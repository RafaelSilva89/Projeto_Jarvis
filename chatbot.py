from flask import Flask, render_template, request, jsonify
from PIL import Image
import pyaudio
import threading
from openai import OpenAI
import wave
import io
import json
import os
from dotenv import load_dotenv
import mss
import base64
from io import BytesIO
import re
import matplotlib.pyplot as plt
import pyautogui
import replicate

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from werkzeug.utils import secure_filename
from langchain.vectorstores.chroma import Chroma
import numpy as np

import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Variáveis globais adicionais
pergunta_num = 0
session_question_count = 0

# Template do prompt para o RAG
PROMPT_TEMPLATE_PDF = """Use o contexto abaixo para responder à pergunta do usuário.
Se você não souber a resposta, apenas diga: Desculpe, não encontrei informações relevantes nos documentos carregados. Não tente inventar uma resposta.

Contexto: {context}

Pergunta: {question}

Resposta:"""

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Configurações iniciais
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

load_dotenv()

# Configurações de áudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Inicialização do cliente OpenAI
client = OpenAI()

class ChatBot:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.chat_history = []
        self.settings = self.load_settings()
        self.text_chunks = []
        self.vector_store = None
        
    def load_settings(self):
        settings_file = 'settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as file:
                return json.load(file)
        return {
            "selected_voice": "alloy",
            "hear_response": True
        }

    def transcribe_audio(self, audio_file):
        try:
            audio_file.seek(0)
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return response.text
        except Exception as e:
            print("Error during transcription:", e)
            return ""

    def get_response(self, user_message):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """You are a helpful assistant.
                        Your answer must be a JSON.
                        If it was a regular question, the type is 'normal'.
                        If it was a regular ask, the type is 'pdf'.
                        If it was a regular ask, the type is 'ctb'.
                        If the user ask to click or to point to something, the type is 'click'.
                        The click content must be always in english starting with 'point to the...'
                        If the user ask about the screen or image, the type is 'image' and pass the question as content."""},
                    {"role": "assistant", "content": "\n".join(self.chat_history)},
                    {"role": "user", "content": user_message}
                ]
            )

            json_response = json.loads(response.choices[0].message.content)
            if json_response.get('type') == 'normal':
                return {'type': 'normal', 'content': json_response.get('content')}
            elif json_response.get('type') == 'pdf':
                rag_response = self.get_rag_response(json_response.get('content'))
                # Extrair apenas o conteúdo da resposta do RAG - PDF
                if isinstance(rag_response, dict):
                    return {'type': 'pdf', 'content': rag_response.get('content', '')}
                return {'type': 'pdf', 'content': str(rag_response)}
            elif json_response.get('type') == 'ctb':
                rag_response = self.get_ragctb_response(json_response.get('content'))
                # Extrair apenas o conteúdo da resposta do RAG - CTB
                if isinstance(rag_response, dict):
                    return {'type': 'ctb', 'content': rag_response.get('content', '')}
                return {'type': 'ctb', 'content': str(rag_response)}
            elif json_response.get('type') == 'image':
                resposta = self.ler_tela(json_response.get('content'))
                return {'type': 'image', 'content': resposta}
            else:
                self.click_on(json_response.get('content'))
                return {'type': 'click', 'content': json_response.get('content')}

        except Exception as e:
            return {'type': 'error', 'content': f"Sorry, I couldn't get a response. Error: {e}"}

    def get_rag_response(self, question):
        if not self.vector_store:
            self.vector_store = get_vector_store(self.text_chunks, os.getenv('OPENAI_API_KEY'))
        
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=5)
        print_formatted_results(results)
        
        if len(results) == 0 or results[0][1] < 0.3:
            return {'type': 'pdf', 'content': "Desculpe, não encontrei informações relevantes nos documentos carregados."}
        
        context = "\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_PDF)
        prompt = prompt_template.format(context=context, question=question)
        
        completion = client.chat.completions.create(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        
        return {'type': 'pdf', 'content': completion.choices[0].message.content}
    
#######################################################################################################
    def get_ragctb_response(self, question):

        # Caminho para a base de dados do Chroma
        CHROMA_PATH_CTB = "chromactb"

        api_key = os.getenv('OPENAI_API_KEY')
        # Inicialização do Chroma e do modelo de embedding
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=CHROMA_PATH_CTB, embedding_function=embedding_function)

        # Pesquisar no banco de dados
        results = db.similarity_search_with_relevance_scores(question, k=5)
        print("Resultados de relevância:", results)
        print_formatted_results(results)
        
        # Verifique se há resultados relevantes
        if len(results) == 0 or results[0][1] < 0.7:
            print("Nenhum resultado relevante encontrado ou abaixo do limiar de 0.3.")
            return {'type': 'ctb', 'content': "Desculpe, não encontrei informações relevantes nos documentos carregados."}
        
        # Concatenar o conteúdo dos documentos relevantes
        conteudo = "\n\n".join([doc.page_content for doc, _score in results])
        print("Conteúdo extraído para o contexto:", conteudo)
        
        # Verificar se `conteudo` está vazio
        if not conteudo.strip():
            print("Conteúdo vazio após a concatenação.")
            return {'type': 'ctb', 'content': "Desculpe, o contexto está vazio e não é possível gerar uma resposta."}
        
        # Estrutura do prompt com o contexto preenchido
        prompt_template = """
        Human: Use o contexto abaixo para responder à pergunta do usuário. Se você não souber a resposta, apenas diga:
        "Desculpe, não encontrei informações relevantes nos documentos carregados. Não tente inventar uma resposta."
        
        Contexto: {context}
        
        Pergunta: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(context=conteudo, question=question)
        print("Conteúdo do prompt final:", prompt)
        
        try:
            completion = client.chat.completions.create(
                temperature=0.5,
                model="gpt-4o-mini",
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
            )
            
            response_content = completion.choices[0].message.content
            print(f"Resposta gerada: {response_content}")
            
            return {'type': 'ctb', 'content': response_content}
            
        except Exception as e:
            print(f"Erro na chamada da API: {str(e)}")
            return {'type': 'ctb', 'content': "Ocorreu um erro ao processar sua pergunta."}
    

#######################################################################################################

    def ler_tela(self, message):
        image = self.capture_and_show_image_from_second_monitor(1920, 1080)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    def click_on(self, click_this):
        #os.environ["REPLICATE_API_TOKEN"] = "TOKEN"
        #api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

        segundo_monitor_pixels = 1920
        image = self.capture_and_show_image_from_second_monitor(1920, 1080)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        output = replicate.run(
            "zsxkib/molmo-7b:76ebd700864218a4ca97ac1ccff068be7222272859f9ea2ae1dd4ac073fa8de8",
            input={
                "text": click_this,
                "image": f"data:image/png;base64,{base64_image}",
                "top_k": 100,
                "top_p": 1,
                "temperature": 1,
                "length_penalty": 1,
                "max_new_tokens": 1000
            }
        )

        pattern = r'x\d*="([\d.]+)" y\d*="([\d.]+)"'
        matches = re.findall(pattern, output)
        coordinates = [(float(x), float(y)) for x, y in matches]

        width, height = image.size
        x_coords = [x / 100 * width for x, y in coordinates]
        y_coords = [y / 100 * height for x, y in coordinates]

        for x, y in zip(x_coords, y_coords):
            pyautogui.moveTo(x + segundo_monitor_pixels, y, duration=0.5)
            pyautogui.click()

        return "Click action completed"

    def capture_and_show_image_from_second_monitor(self, width=400, height=400):
        with mss.mss() as sct:
            monitors = sct.monitors
            if len(monitors) < 2:
                raise ValueError("No second monitor detected.")
            
            second_monitor = monitors[2]
            capture_region = {
                "top": second_monitor["top"],
                "left": second_monitor["left"],
                "width": width,
                "height": height
            }
            
            screenshot = sct.grab(capture_region)
            return Image.frombytes("RGB", screenshot.size, screenshot.rgb)

# Add the new functions and route at the module level
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    documents = [Document(page_content=text)]
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def get_vector_store(text_chunks, api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Erro ao inicializar os embeddings: {str(e)}")
        raise

# Função para formatar resultados de busca
def print_formatted_results(results):
    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        result = {
            "number": i,
            "length": len(doc.page_content),
            "score": score,
            "content": doc.page_content
        }
        formatted_results.append(result)

    # Save results to a JSON file
    with open('static/pdf_results.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('file')
    pdf_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_files.append(file_path)
    
    if pdf_files:
        try:
            text = get_pdf_text(pdf_files)
            chatbot.text_chunks = get_text_chunks(text)
            # Inicializa o vector store
            chatbot.vector_store = get_vector_store(chatbot.text_chunks, os.getenv('OPENAI_API_KEY'))
            return jsonify({
                'message': 'Arquivos PDF processados com sucesso',
                'files': [os.path.basename(f) for f in pdf_files]
            })
        except Exception as e:
            return jsonify({'error': f'Erro ao processar PDFs: {str(e)}'})
    else:
        return jsonify({'error': 'Nenhum arquivo PDF válido foi selecionado'})
    
# Rota para mostrar as informações do PDF
@app.route('/pdf_info')
def pdf_info():
    return render_template('pdf_info.html')
            
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        #voice = data.get('voice', 'alloy')
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Usando sempre 'alloy'
            input=text,
            response_format="wav"
        )
        
        audio_data = response.content
        audio_file = io.BytesIO(audio_data)
        audio_file.seek(0)
        
        # Play the WAV audio using PyAudio
        wf = wave.open(audio_file, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return jsonify({
            'status': 'success',
            'audio': audio_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    chatbot.settings.update(data)
    with open('settings.json', 'w') as file:
        json.dump(chatbot.settings, file)
    return jsonify({'status': 'success'})

# Instância global do chatbot
chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    chatbot.chat_history.append(f"You: {message}")
    response = chatbot.get_response(message)
    chatbot.chat_history.append(f"Bot: {response['content']}")
    
    return jsonify(response)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if not chatbot.is_recording:
        chatbot.is_recording = True
        chatbot.frames = []
        
        def record_audio():
            audio_stream = chatbot.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            while chatbot.is_recording:
                data = audio_stream.read(CHUNK)
                chatbot.frames.append(data)
            
            audio_stream.stop_stream()
            audio_stream.close()
        
        threading.Thread(target=record_audio).start()
        return jsonify({'status': 'Recording started'})
    return jsonify({'error': 'Already recording'}), 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    if chatbot.is_recording:
        chatbot.is_recording = False
        
        # Convert recorded audio to bytes
        audio_data = b''.join(chatbot.frames)
        audio_file = io.BytesIO()
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(chatbot.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        audio_file.name = "output.wav"
        
        # Transcribe audio and get response
        transcript = chatbot.transcribe_audio(audio_file)
        if transcript:
            chatbot.chat_history.append(f"You: {transcript}")
            response = chatbot.get_response(transcript)
            chatbot.chat_history.append(f"Bot: {response['content']}")
            return jsonify({
                'transcript': transcript,
                'response': response
            })
    
    return jsonify({'error': 'Not recording'}), 400

if __name__ == '__main__':
    app.run(debug=True)