import pandas as pd
import os
import time
from openai import OpenAI
from tqdm.auto import tqdm
from Eugene_v2.src.utils import read_dataframe, write_dataframe


def extract_primary_activities(df, input_col, output_col, client, model_name="gpt-4o-mini"):
    """
    Extract primary economic activities from company descriptions using OpenAI API.

    Args:
        df: Input dataframe
        input_col: Column containing company descriptions
        output_col: Column name for extracted activities
        client: OpenAI client
        model_name: OpenAI model to use (default: gpt-4o-mini)

    Returns:
        DataFrame with new output_col containing extracted activities
    """
    df = df.copy()

    system_prompt = """You are an expert at identifying economic activities from company descriptions. Extract primary and/or secondary revenue-generating activities based on the company description provided."""

    user_prompt_template = """Identify and extract primary economic activities from this company description.

Rules:
- Base your decision ONLY on the company description
- Focus on revenue-generating activities
- Leave empty if unsure (>0.8 confidence required)
- Extract maximum 2 activities, minimum 0
- Output only the activities, nothing else

Company Description:
{description}"""

    results = []

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting activities"):
        try:
            description = str(row[input_col]) if pd.notna(row[input_col]) else ""

            if not description.strip():
                results.append("")
                continue

            # Create chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(description=description[:2000])}
            ]

            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()
            results.append(result)

            # Rate limiting - brief pause between requests
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            results.append("")

    df[output_col] = results
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Process only first N rows")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    args = parser.parse_args()

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    print(f"Using OpenAI model: {args.model}")

    # Load data
    print("\nLoading data...")
    df = read_dataframe("results_data/SE_with_base_min_5_others_inferenced_BASELINE.xlsx")
    print(f"Loaded {len(df)} rows")

    # Sample if requested
    if args.sample:
        print(f"Sampling first {args.sample} rows for testing...")
        df = df.head(args.sample)

    # Extract activities
    print("\nExtracting primary activities...")
    result_df = extract_primary_activities(
        df.copy(),
        input_col="About the Company_predicted",
        output_col="primary_activities",
        client=client,
        model_name=args.model
    )

    # Save results
    output_file = "results_data/SE_with_primary_activities.xlsx"
    print(f"\nSaving results to {output_file}...")
    write_dataframe(result_df, output_file)
    print("Done!")
