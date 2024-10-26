import os
import traceback
import time
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import mimetypes
import magic

import openai

load_dotenv()

CHROMA_PATH = "chromactb"
DATA_PATH = "data"
CHECKPOINT_FILE = "checkpoint.json"
BATCH_SIZE = 100
PERSIST_FREQUENCY = 25

def detect_file_type(file_path):
    try:
        mime = magic.from_file(file_path, mime=True)
    except magic.MagicException:
        mime, _ = mimetypes.guess_type(file_path)
    return mime

def main():
    global db  # Definir 'db' como global para o teste
    try:
        documents = load_documents()
        print(f"Total de documentos carregados: {len(documents)}")
        if documents:
            chunks = split_text(documents)
            process_chunks(chunks)
        test_database()
    except Exception as e:
        print(f"Ocorreu um erro durante a execução do script:")
        print(str(e))
        print("\nStack trace completo:")
        print(traceback.format_exc())
    finally:
        print("Script concluído. Pressione Enter para sair.")
        input()

def load_documents():
    directory_path = DATA_PATH
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)

    documents = []
    try:
        documents = loader.load()  # Carregar documentos diretamente
        if not documents:
            print("Nenhum documento carregado.")
    except Exception as e:
        print(f"Erro ao carregar documentos: {e}")

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    # Garantir que documents contém instâncias com o atributo 'page_content'
    if documents and isinstance(documents[0], dict):
        documents = [Document(page_content=doc.get('content', '')) for doc in documents]

    chunks = text_splitter.split_documents(documents)
    print(f"Criados {len(chunks)} chunks de texto.")
    return chunks

def process_chunks(chunks):
    global db  # Definir 'db' como global para o teste
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("A chave API do OpenAI não foi encontrada nas variáveis de ambiente.")

    print(f"Iniciando o processo de salvar no Chroma...")
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    print("Embedding function criada com sucesso.")

    last_processed_chunk = load_checkpoint()
    
    try:
        if os.path.exists(CHROMA_PATH):
            print(f"Atualizando banco de dados existente em {CHROMA_PATH}")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        else:
            print(f"Criando novo banco de dados em {CHROMA_PATH}")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        for i in range(last_processed_chunk, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            process_batch(db, batch, i, chunks)

        print("Persistindo os dados finais...")
        db.persist()
        print(f"Salvos {len(chunks)} chunks em {CHROMA_PATH}.")

        # Verificação final
        collection = db._collection
        print(f"Número total de documentos no banco de dados: {collection.count()}")
        
    except Exception as e:
        print(f"Erro ao salvar no Chroma: {str(e)}")
        print(traceback.format_exc())
        raise

def process_batch(db, batch, start_index, chunks):
    for i, chunk in enumerate(batch):
        overall_index = start_index + i
        if overall_index % 10 == 0:
            print(f"Processando chunk {overall_index+1}/{len(chunks)}...")
        try:
            start_time = time.time()
            db.add_documents([chunk])
            end_time = time.time()
            print(f"Tempo para processar chunk {overall_index+1}: {end_time - start_time:.2f} segundos")
        except openai.error.OpenAIError as oe:
            print(f"Erro da API OpenAI ao processar chunk {overall_index+1}: {str(oe)}")
            # Implementar lógica de retry aqui, se necessário
        except Exception as e:
            print(f"Erro ao processar chunk {overall_index+1}: {str(e)}")
            raise
        
        if (overall_index + 1) % PERSIST_FREQUENCY == 0:
            print("Persistindo dados intermediários...")
            db.persist()
            save_checkpoint(overall_index + 1)
            print("Dados intermediários persistidos.")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)['last_processed_chunk']
    return 0

def save_checkpoint(last_processed_chunk):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'last_processed_chunk': last_processed_chunk}, f)

def test_database():
    global db  # Definir 'db' como global para o teste
    print("\nTestando o banco de dados:")
    
    try:
        # Teste 1: Verificar o número de documentos
        print(f"Número de documentos no banco: {db._collection.count()}")
        
        # Teste 2: Recuperar alguns documentos aleatórios
        results = db.similarity_search("", k=2)
        print("\nAlguns documentos aleatórios do banco:")
        for doc in results:
            print(f"- {doc.page_content[:100]}...")
        
        # Teste 3: Pergunta específica
        query = "Não chegou a minha CNH, o que eu posso fazer?"
        results = db.similarity_search(query, k=2)
        
        print(f"\nResultados para a pergunta: '{query}'")
        for doc in results:
            print(f"- {doc.page_content[:200]}...")
        
        # Teste 4: Usar o modelo de linguagem para responder à pergunta
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "Responda à seguinte pergunta com base no contexto fornecido:\n\nPergunta: {query}\n\nContexto: {context}\n\nResposta:"
        )
        
        context = "\n".join([doc.page_content for doc in results])
        response = llm.invoke(prompt.format(query=query, context=context))
        
        print(f"\nResposta gerada pelo modelo:")
        print(response.content)
    except Exception as e:
        print(f"Erro durante o teste do banco de dados: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
