import argparse
import json
import sys
from vecdb.client.sdk import VecDBClient


def main():
    parser = argparse.ArgumentParser(description="vdb — VecDB Command-Line Interface")
    parser.add_argument("--url", default="http://localhost:8000", help="VecDB Server Base URL")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: create-collection
    cmd_create = subparsers.add_parser("create-collection", help="Create a new collection")
    cmd_create.add_argument("--name", required=True, help="Collection name")
    cmd_create.add_argument("--dimension", type=int, required=True, help="Vector dimension")
    cmd_create.add_argument("--metric", default="cosine", help="Distance metric")

    # Command: insert
    cmd_insert = subparsers.add_parser("insert", help="Insert vector into collection")
    cmd_insert.add_argument("--collection", required=True, help="Collection name")
    cmd_insert.add_argument("--id", required=True, help="Vector ID")
    cmd_insert.add_argument("--vector", required=True, help="JSON list string of float vector, e.g. '[1.0, 2.0]'")
    cmd_insert.add_argument("--payload", default=None, help="JSON payload string, e.g. '{\"category\": \"A\"}'")

    # Command: search
    cmd_search = subparsers.add_parser("search", help="Execute vector similarity search")
    cmd_search.add_argument("--collection", required=True, help="Collection name")
    cmd_search.add_argument("--query", required=True, help="JSON query vector string")
    cmd_search.add_argument("--k", type=int, default=10, help="Top-k count")
    cmd_search.add_argument("--filter", default=None, help="JSON filter string")
    cmd_search.add_argument("--index", default="hnsw", help="Index type: flat, ivf, hnsw")

    # Command: stats
    cmd_stats = subparsers.add_parser("stats", help="Get collection statistics")
    cmd_stats.add_argument("--collection", required=True, help="Collection name")

    # Command: snapshot
    cmd_snap = subparsers.add_parser("snapshot", help="Trigger snapshot for collection")
    cmd_snap.add_argument("--collection", required=True, help="Collection name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = VecDBClient(base_url=args.url)

    try:
        if args.command == "create-collection":
            res = client.create_collection(args.name, args.dimension, args.metric)
            print(json.dumps(res, indent=2))
        elif args.command == "insert":
            vector = json.loads(args.vector)
            payload = json.loads(args.payload) if args.payload else None
            res = client.insert(args.collection, args.id, vector, payload)
            print(json.dumps(res, indent=2))
        elif args.command == "search":
            query = json.loads(args.query)
            filter_q = json.loads(args.filter) if args.filter else None
            res = client.search(args.collection, query, k=args.k, filter_query=filter_q, index_type=args.index)
            print(json.dumps(res, indent=2))
        elif args.command == "stats":
            res = client.get_stats(args.collection)
            print(json.dumps(res, indent=2))
        elif args.command == "snapshot":
            res = client.snapshot(args.collection)
            print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
