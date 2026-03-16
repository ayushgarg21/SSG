import pandas as pd
import os
import time
from openai import OpenAI
from tqdm.auto import tqdm
from Eugene_v2.src.utils import read_dataframe, write_dataframe

# Prompt templates used by all backends
SYSTEM_PROMPT = """You are an expert at identifying economic activities from company descriptions. Extract primary and/or secondary revenue-generating activities based on the company description provided. You only output the activites, no other addtional text."""

USER_PROMPT_TEMPLATE = """Identify and extract primary economic activities from this company description.

Rules:
- Base your decision ONLY on the company description
- Focus on revenue-generating activities
- If the industry is explicitly mentioned in the company description, consider evaluating it as a an options
- Leave empty if unsure (>0.75 confidence required)
- Extract maximum 2 activities, minimum 0
- Output only the activities, nothing else, do not include any explanations or additional text
- Output format: activity1[, activity2] (comma-separated, no newline,no numbering bullet)
- Example 1 activity: Banking and financial services
- Example 2 activities: News agency, IT service provider

Company Description:
{description}"""


def extract_with_openai(df, input_col, output_col, client, model_name="gpt-4o-mini"):
    """Extract activities using OpenAI API"""
    df = df.copy()
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description[:2000])}
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
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            results.append("")

    df[output_col] = results
    return df


# model_name="qwen2.5:7b-instruct"


def extract_with_ollama(df, input_col, output_col, model_name="llama3:8b-instruct-q8_0"):
    """Extract activities using Ollama with Qwen"""
    import ollama

    df = df.copy()
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting activities (Ollama)"):
        try:
            description = str(row[input_col]) if pd.notna(row[input_col]) else ""

            if not description.strip():
                results.append("")
                continue

            # Create full prompt
            prompt = f"""{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(description=description[:2000])}"""

            # Call Ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.05,
                    "top_p": 1.0
                }
            )

            result = response["message"]["content"].strip()
            # Handle Qwen null responses (returns ">")
            if result == ">":
                result = ""
            results.append(result)

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            results.append("")

    df[output_col] = results
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Process only first N rows")
    parser.add_argument("--backend", default="openai", choices=["openai", "ollama"],
                       help="Backend to use: openai, or ollama (default: openai)")
    parser.add_argument("--model", help="Model name (default depends on backend)")
    args = parser.parse_args()

    # Set default model based on backend
    if args.model is None:
        if args.backend == "openai":
            args.model = "gpt-4o-mini"
        elif args.backend == "ollama":
            args.model = "qwen2.5:7b-instruct"

    print(f"Using backend: {args.backend}")
    print(f"Using model: {args.model}")

    # Load data
    print("\nLoading data...")
    df = read_dataframe("results_data/SE_with_base_min_5_others_inferenced_BASELINE.xlsx")
    print(f"Loaded {len(df)} rows")

    # Sample if requested
    if args.sample:
        print(f"Sampling first {args.sample} rows for testing...")
        df = df.head(args.sample)

    # Extract activities based on backend
    print("\nExtracting primary activities...")

    if args.backend == "openai":
        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)

        result_df = extract_with_openai(
            df.copy(),
            input_col="About the Company_predicted",
            output_col="primary_activities",
            client=client,
            model_name=args.model
        )


    elif args.backend == "ollama":
        print("Using Ollama (make sure Ollama is running: ollama serve)")
        result_df = extract_with_ollama(
            df.copy(),
            input_col="About the Company_predicted",
            output_col="primary_activities",
            model_name=args.model
        )

    # Save results
    output_file = "results_data/SE_with_primary_activities_ollama_v2.xlsx"
    print(f"\nSaving results to {output_file}...")
    write_dataframe(result_df, output_file)
    print("Done!")
