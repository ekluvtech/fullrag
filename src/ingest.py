import argparse
import os
from chunking import build_chunks_from_directory, iter_chunks_from_directory, iter_chunks_with_archive
from ingestion import IngestionPipeline


def main():
	parser = argparse.ArgumentParser(description="Ingest a directory of documents into Qdrant")
	parser.add_argument("--input", required=True, help="Path to directory")
	parser.add_argument("--collection", default=None, help="Target collection name")
	parser.add_argument("--stream", action="store_true", help="Use streaming ingestion (recommended for very large datasets)")
	parser.add_argument("--archive", action="store_true", help="Archive processed files to skip them in future runs")
	parser.add_argument("--archive-dir", default=None, help="Archive directory (default: input_dir/archive)")
	args = parser.parse_args()

	# Set up archive directory
	archive_dir = args.archive_dir
	if args.archive and not archive_dir:
		archive_dir = os.path.join(args.input, "archive")

	pipe = IngestionPipeline()
	if args.stream:
		if args.archive:
			chunk_iter = iter_chunks_with_archive(args.input, archive_dir)
		else:
			chunk_iter = iter_chunks_from_directory(args.input, archive_dir)
		pipe.ingest_stream(chunk_iter, collection=args.collection)
	else:
		chunks = build_chunks_from_directory(args.input, archive_dir)
		pipe.ingest(chunks, collection=args.collection)


if __name__ == "__main__":
	main()
