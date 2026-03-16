import pandas as pd
import os
import json
import time
from openai import OpenAI
from tqdm.auto import tqdm
from Eugene_v2.src.utils import read_dataframe, write_dataframe

JUDGE_PROMPT_TEMPLATE = """You are evaluating extracted primary business activities for industry classification (SSIC).

Company description:
{company_description}

Extracted primary activity:
{primary_activity}

Evaluate on the following criteria (score 1–5 each):

1. Relevance – describes the company's main business activity, not a job function
2. Specificity – concrete enough for SSIC classification
3. Revenue focus – represents a core revenue-generating activity

Respond ONLY in valid JSON:

{{
  "relevance": number,
  "specificity": number,
  "revenue_focus": number,
  "overall_comment": "short explanation"
}}"""


def judge_activities_batch(df, description_col, activity_col, batch_size=20):
    """
    Judge extracted activities using OpenAI in batches.

    Args:
        df: DataFrame with company descriptions and extracted activities
        description_col: Column name containing company descriptions
        activity_col: Column name containing extracted activities
        batch_size: Number of items to process per batch

    Returns:
        DataFrame with added judge scores columns
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    df = df.copy()

    # Initialize result columns
    df['relevance'] = None
    df['specificity'] = None
    df['revenue_focus'] = None
    df['overall_comment'] = None
    df['judge_error'] = None

    # Process in batches
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Judging activities"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        # Process each item in the batch
        for idx, row in batch_df.iterrows():
            try:
                description = str(row[description_col]) if pd.notna(row[description_col]) else ""
                activity = str(row[activity_col]) if pd.notna(row[activity_col]) else ""

                # Skip if activity is empty
                if not activity.strip():
                    df.at[idx, 'judge_error'] = "No activity to judge"
                    continue

                # Create prompt
                prompt = JUDGE_PROMPT_TEMPLATE.format(
                    company_description=description[:1500],  # Limit description length
                    primary_activity=activity[:500]  # Limit activity length
                )

                # Call OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of business activity descriptions for industry classification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}  # Ensure JSON response
                )

                # Parse JSON response
                result_text = response.choices[0].message.content.strip()
                result = json.loads(result_text)

                # Store results
                df.at[idx, 'relevance'] = result.get('relevance')
                df.at[idx, 'specificity'] = result.get('specificity')
                df.at[idx, 'revenue_focus'] = result.get('revenue_focus')
                df.at[idx, 'overall_comment'] = result.get('overall_comment', '')

            except json.JSONDecodeError as e:
                df.at[idx, 'judge_error'] = f"JSON parse error: {str(e)}"
                print(f"\nJSON parse error at index {idx}: {e}")
            except Exception as e:
                df.at[idx, 'judge_error'] = f"Error: {str(e)}"
                print(f"\nError at index {idx}: {e}")

        # Brief pause between batches to avoid rate limits
        if batch_end < len(df):
            time.sleep(0.5)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Judge extracted primary activities using OpenAI")
    parser.add_argument("--input", required=True, help="Input Excel file (output from extraction.py)")
    parser.add_argument("--description_col", default="About the Company_predicted",
                       help="Column containing company descriptions")
    parser.add_argument("--activity_col", default="primary_activities",
                       help="Column containing extracted activities")
    parser.add_argument("--output", help="Output Excel file (default: input_file with _judged suffix)")
    parser.add_argument("--batch_size", type=int, default=20,
                       help="Batch size for processing (default: 20)")
    parser.add_argument("--sample", type=int, help="Process only first N rows (for testing)")
    args = parser.parse_args()

    # Load input file
    print(f"Loading data from {args.input}...")
    df = read_dataframe(args.input)
    print(f"Loaded {len(df)} rows")

    # Sample if requested
    if args.sample:
        print(f"Sampling first {args.sample} rows for testing...")
        df = df.head(args.sample)

    # Check columns exist
    if args.description_col not in df.columns:
        raise ValueError(f"Description column '{args.description_col}' not found in input file")
    if args.activity_col not in df.columns:
        raise ValueError(f"Activity column '{args.activity_col}' not found in input file")

    # Run judging
    print(f"\nJudging activities (batch size: {args.batch_size})...")
    result_df = judge_activities_batch(
        df,
        description_col=args.description_col,
        activity_col=args.activity_col,
        batch_size=args.batch_size
    )

    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        input_base, input_ext = os.path.splitext(args.input)
        output_file = f"{input_base}_judged{input_ext}"

    # Save results
    print(f"\nSaving results to {output_file}...")
    write_dataframe(result_df, output_file)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Count non-null scores
    valid_scores = result_df['relevance'].notna().sum()
    errors = result_df['judge_error'].notna().sum()

    print(f"Successfully judged: {valid_scores}/{len(result_df)}")
    print(f"Errors: {errors}/{len(result_df)}")

    if valid_scores > 0:
        print("\nAverage Scores:")
        print(f"  Relevance:     {result_df['relevance'].mean():.2f}")
        print(f"  Specificity:   {result_df['specificity'].mean():.2f}")
        print(f"  Revenue Focus: {result_df['revenue_focus'].mean():.2f}")

    print(f"\nOutput saved to: {output_file}")
    print("Done!")
