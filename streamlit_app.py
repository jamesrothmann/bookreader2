import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
from os.path import exists
import numpy as np
import math
from io import StringIO
import openai
import pandas as pd
import math
import urllib.request
import base64
#from gsheetsdb import connect
import gspread
from google.oauth2 import service_account

def authenticate_and_connect(sheet_id):
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["google"], scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    return sheet

@st.cache_resource
def download_file():
    url = "https://drive.google.com/uc?export=download&id=1lXc7npYfbUW78erBKhiEjRN0AQ6Rk4lD"
    #url = "https://drive.google.com/uc?export=download&id=1eaypC-XqCGKn56VZRIQdlwsYHwIbQUJr" #Zero to One  - Peter Thiel
    #url = "https://drive.google.com/uc?export=download&id=1e_bneSaNGhY77Nt07RhTjcMekvwHRGjS" #David Senra Podcast Transcripts
    path = "file.json"

    # Use urllib.request.urlretrieve to download the file from the given URL
    urllib.request.urlretrieve(url, path)

    # Return the path to the downloaded file
    return path

# Download the file and get the path to the downloaded file
path = download_file()


# Set up the OpenAI API key
openai.api_key = st.secrets["api_secret"]

prompt_text = "You are an summary bot."

# Define the OpenAI function
def openaiapi(input_text):
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": input_text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


@st.cache(allow_output_mutation=True)
def load_model():
    model1 = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    return model1

model = load_model()

def get_embeddings(texts):
    if type(texts) == str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)
  
def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return (values['chapters'], np.array(values['embeddings']))


def read_epub(book_path, json_path, preview_mode, first_chapter, last_chapter):
    chapters = get_chapters(book_path, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return (chapters, None)
    print('Generating embeddings for chapters {}-{} in "{}"\n'.format(first_chapter, last_chapter, book_path))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except:
        print('Failed to save embeddings to "{}"'.format(json_path))
    return (chapters, embeddings)

def process_file(path, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    values = None
    if path[-4:] == 'json':
        values = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            values = read_json(json_path)
        else:
            values = read_epub(path, json_path, preview_mode, first_chapter, last_chapter) 
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')        
    return values
  
chapters, embeddings = process_file(path)
  
def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paras'])
        if index < paras_len:
            return chapter['paras'][index], chapter['title'], index
        index -= paras_len
    return None

def search(query, embeddings, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    search_results = []
    for index in results:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        search_results.append(f"{para} ({title}, para {para_no})")

    return search_results



st.title("Streamlit App for Ebook Search and OpenAI Integration")
book_podcast_name = st.text_input("A) Input box for a book/podcast name")
#embeddings_link = st.text_input("B) Input for Link to the JSON Embeddings")

initial_questions = st.text_area("C) Input for List of Initial Questiuons (One per Line)").split("\n")
num_follow_up_questions = st.slider("D) Amount of follow-up questions", 1, 10)
submit_button = st.button("Submit")

raw_api_responses = [] 

def parse_content_to_dataframe(json_content):
    data = json.loads(json_content)
    return pd.DataFrame.from_records(data)

def convert_table_to_json(table_text):
    lines = table_text.strip().split('\n')
    header = lines[0].split('|')[1:-1]
    data = [dict(zip(header, line.split('|')[1:-1])) for line in lines[1:]]
    return json.dumps(data)

def append_to_dataframe(df, data):
    data_string = StringIO(data)
    new_df = pd.read_csv(data_string, sep='|')
    return pd.concat([df, new_df], ignore_index=True)

def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=2)

def get_download_link(filename, text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    

def append_dataframe_to_gsheet(df, sheet_id):
    sheet = authenticate_and_connect(sheet_id)
    worksheet = sheet.get_worksheet(0)  # Assuming you want to append to the first sheet
    print("Appending data...")  # Debugging print statement

    for index, row in df.iterrows():
        print(f"Appending row {index + 1}...")  # Debugging print statement
        row_data = list(row.values)
        worksheet.append_row(row_data)
    print("Data appended successfully")  # Debugging print statement
        
  

if submit_button:
    columns = ["Question", "Chapter", "Quote", "30-word_ummary", "Tag_1", "Tag_2", "Tag_3", "Tag_4", "Tag_5", "7_Word_Problem_Statement", "Emotion_Triggered", "Content_type"]
    results_df = pd.DataFrame(columns=columns)

    for question in initial_questions:
        search_results = search(question, embeddings)
        
        prompt1 = '''The below are extracts based on a semantic search from a book or a podcast transcript with the name '{}' in response to the question '{}'. \\nI want you to extract lessons or principles or secrets for success, building wealth, business advice and/or investing in a table in the following format in the following JSON format: 

        [
            {{
                "Question": "How does Amazon maintain customer obsession as a core principle throughout its rapid growth and expansion?",
                "Chapter": "Jeff Bezos - Invent and Wander",
                "Quote": "One thing I love about customers is that they are divinely discontent. Their expectations are never static—they go up.",
                "30_word_summary": "Customers' expectations are never static; they go up.",
                "Tag_1": "Customer Obsession",
                "Tag_2": "Innovation",
                "Tag_3": "Constant Improvement",
                "Tag_4": "Motivation",
                "Tag_5": "Core Principles",
                "7_word_problem_statement": "Customers' expectations are never static.",
                "Emotion_tigger": "WOW – That’s amazing",
                "Content_type": "Counter-Intuitive"
            }}
          ]

        Paragraph/Sentence/Quote - This must be an extract from the text. It must be either counter-intuitive (Not how I expected the world to work) or counter-narrative (Not how I was told it works), or be elegantly articulated (wish that I could have said it like that). 

        Emotion Triggered: | LOL – That’s so funny| WTF – That pisses me off | AWW – That’s so cute | WOW – That’s amazing | NSFW – That’s Crazy| OHHHH – Now I Get it | FINALLY – someone said what I feel| YAY – That’s great news|

        Content Type: Counter-intuitive, Counter-Narrative, or Elegant Articulation

        Extract:
        '''.format(book_podcast_name, question)

        search_results_text = "\n".join(search_results)
        prompt1_with_results = "{}\n{}".format(prompt1, search_results_text)

        
        max_retries = 4
        retries = 0

        while retries <= max_retries:
            try:
                api_response = openaiapi(prompt1_with_results)
                raw_api_responses.append(api_response.choices[0].to_dict())  # Save the raw JSON response
                text_response = api_response.choices[0].message['content'].strip()  # Extract the text content

                json_content = convert_table_to_json(text_response)
                df = parse_content_to_dataframe(json_content)

                google_sheet_id = '1DRUNh6JPLDuTtrpmyGCefMQX_Ipfx-PhxPKXj6eIcTk'
                append_dataframe_to_gsheet(df, google_sheet_id)
                st.write(df)
                print(df)
                print(text_response)
                Print(json_content)
            
                break  # Exit the loop if the API call was successful
            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                if retries > max_retries:
                    st.write("Max retries reached. Unable to get a valid response from the API.")
                else:
                    st.write(f"Attempt {retries}/{max_retries}: Retrying...")

    st.success("Task Completed")
        # ... (existing code) ...

    # Save the raw API responses to a text file
    output_filename = 'raw_api_responses.json'
    save_responses_to_file(raw_api_responses, output_filename)

    # Display the download link
    with open(output_filename, 'r') as f:
        text = f.read()
        download_link = get_download_link(output_filename, text)
        st.markdown(download_link, unsafe_allow_html=True)
