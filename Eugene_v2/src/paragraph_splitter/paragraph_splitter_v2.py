# paragraph_splitter.py - v2 - try using chonkie
import pandas as pd
from tqdm import tqdm
tqdm.pandas()  # enable progress_apply

import argparse
from chonkie import SemanticChunker,NeuralChunker
from chonkie import SlumberChunker
from chonkie.genie import GeminiGenie,OpenAIGenie

from transformers import AutoTokenizer

from Eugene.ml_classify.jobert_utils import preprocess_line_breaks,split_into_sentences
from Eugene_v2.src.utils import read_dataframe, write_dataframe


def df_to_paragraphs(df: pd.DataFrame, html_col: str = "raw_html", output_col: str = "neu_chunks"):
    sem_chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",  # Default model
        threshold=0.95,                               # Similarity threshold (0-1)
        chunk_size=2048,                             # Maximum tokens per chunk
        similarity_window=3,                         # Window for similarity calculation
        skip_window=0,                                # Skip-and-merge window (0=disabled)
        delim=[". ", "! ", "? ", "\n\n"],  # Custom sentence delimiters

    )

    # neu_chunker = NeuralChunker(
    #     model="mirth/chonky_modernbert_base_1",  # Default model
    #     device_map="mps",                        # Device to run the model on ('cpu', 'cuda', etc.)
    #     min_characters_per_chunk=10,             # Minimum characters for a chunk
    # )


  
    neu_chunker = NeuralChunker(
    model="mirth/chonky_modernbert_base_1",  # Default model
    device_map="mps",                        # Device to run the model on ('cpu', 'cuda', etc.)
    min_characters_per_chunk=5           # Minimum characters for a chunk
)

    genie = OpenAIGenie()

    slum_chunker = SlumberChunker(
        genie=genie,                        # Genie interface to use
        tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
        chunk_size=1024,                    # Maximum chunk size
        candidate_size=128,                 # How many tokens Genie looks at for potential splits
        min_characters_per_chunk=2,        # Minimum number of characters per chunk
        verbose=True                        # See the progress bar for the chunking process
    )

    def get_chunks(text,chunker,lists=False):
        try:
            chunks = chunker.chunk(text)
            if lists:
                return [ c.text for c in chunks]
            else :
                return [{"text": c.text, "tokens": c.token_count} for c in chunks]
        except (IndexError, Exception) as e:
            # Handle chunking errors gracefully
            print(f"\nWarning: Chunking failed with error: {e}")
            print(f"Returning original text as single chunk")
            if lists:
                return [text]
            else:
                return [{"text": text, "tokens": len(text)}]
     

# Apply to dataframe
    df[html_col] = df[html_col].progress_apply(preprocess_line_breaks)
    # Neural chunks
    df[output_col] = df[html_col].progress_apply(lambda x: get_chunks(x, neu_chunker,lists=True))
    df["num"+output_col] = df[output_col].apply(len)
    df.to_csv("results_data/se_neural_only_large_min2.csv")

    # df["slum_chunks"] = df[html_col].progress_apply(lambda x: get_chunks(x, slum_chunker,lists=True))
    # df["num_slum_chunks"] = df["slum_chunks"].apply(len)

    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Input file (.csv, .xlsx, .xls, .parquet, .pq)")
    parser.add_argument("--html_col", default="raw_html")
    parser.add_argument("--output_file", required=True, help="Output file (.csv, .xlsx, .xls, .parquet, .pq)")
    parser.add_argument("--sample", type=int, help="Process only first N rows (for testing)")
    args = parser.parse_args()

    df = read_dataframe(args.input_file)

    # Sample if requested
    if args.sample:
        print(f"Processing sample of {args.sample} rows (out of {len(df)} total)")
        df = df.head(args.sample)

    df = df_to_paragraphs(df, html_col=args.html_col)
    write_dataframe(df, args.output_file)
    print("Wrote paragraphized dataframe to", args.output_file)

      # Explode the chunks into separate rows
    df = df.explode('neu_chunks', ignore_index=True)
    write_dataframe(df, "exploded_base_min_5.xlsx")

