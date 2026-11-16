from src.embeddings import create_embedding_model
from src.rag import ask


def print_header():
    print("=" * 60)
    print("🤖 MINI RAG AI ASSISTANT")
    print("=" * 60)
    print("Ask questions about your documents.")
    print("Commands:")
    print("  /sources  → show sources of the last answer")
    print("  /clear    → clear conversation")
    print("  /exit     → quit")
    print("=" * 60)


def print_sources(results):
    if not results:
        print("\n📚 Sources: None")
        return

    print("\n📚 Sources:")

    seen = set()

    for result in results:
        source = result["source"]

        if source not in seen:
            print(
                f"  • {source} "
                f"(similarity: {result['score']:.3f})"
            )
            seen.add(source)


def main():

    print_header()

    print("\n🔄 Loading embedding model...")

    embedding_model = create_embedding_model()

    print("✅ Embedding model ready.")

    conversation = []
    last_results = []

    while True:

        print()

        question = input("🧑 You: ").strip()

        # -------------------------
        # Commands
        # -------------------------

        if question.lower() == "/exit":
            print("\n👋 Goodbye!")
            break

        if question.lower() == "/clear":
            conversation.clear()
            last_results.clear()

            print("\n🧹 Conversation cleared.")
            continue

        if question.lower() == "/sources":
            print_sources(last_results)
            continue

        if not question:
            continue

        # -------------------------
        # RAG
        # -------------------------

        try:

            print("\n🔎 Searching documents...")

            answer, results = ask(
                question,
                embedding_model,
                top_k=3
            )

            last_results = results

            # Save conversation
            conversation.append({
                "user": question,
                "assistant": answer
            })

            # -------------------------
            # Answer
            # -------------------------

            print("\n🤖 Assistant:")
            print(answer)

            # -------------------------
            # Sources
            # -------------------------

            print_sources(results)

        except Exception as error:

            print("\n❌ Error:")
            print(error)


if __name__ == "__main__":
    main()