import typer

from lit2vec.cli.commands import bm25, embeddings, faiss, pipeline, prepare, sql_index
from lit2vec.cli.console import print_banner

app = typer.Typer(
    name="lit2vec",
    help="Embedding and search-index toolkit for biomedical literature corpora.",
    rich_markup_mode="rich",
    add_completion=False,
)

# Long names
app.command(name="generate-embeddings")(embeddings.generate_embeddings)
app.command(name="make-sql-index")(sql_index.make_sql_index)
app.command(name="make-bm25-index")(bm25.make_bm25_index)
app.command(name="make-faiss-index")(faiss.make_faiss_index)
app.command(name="pipeline")(pipeline.run_pipeline)
app.command(name="prepare")(prepare.prepare)

# Short aliases
app.command(name="embed", hidden=True)(embeddings.generate_embeddings)
app.command(name="sql", hidden=True)(sql_index.make_sql_index)
app.command(name="bm25", hidden=True)(bm25.make_bm25_index)
app.command(name="faiss", hidden=True)(faiss.make_faiss_index)
app.command(name="run", hidden=True)(pipeline.run_pipeline)
app.command(name="prep", hidden=True)(prepare.prepare)


@app.callback()
def callback():
    """Lit2Vec CLI — embeddings, FAISS, BM25, and metadata indexing."""
    print_banner()


def cli():
    app()


if __name__ == "__main__":
    cli()
