import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
import google.generativeai as genai
import re
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import pandas as pd


from groq import Groq

client = Groq(
    api_key="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",
)

# Cấu hình Google API
os.environ["COHERE_API_KEY"] = "2MW0YheImX1HofZWdKcyyFnBVtrg19d2EV0GjJnI"
genai.configure(api_key="AIzaSyB3j7vAOJBL4MnWPk8VJJM1Yg33YTZEBv0")
os.environ["GOOGLE_API_KEY"] = "AIzaSyB3j7vAOJBL4MnWPk8VJJM1Yg33YTZEBv0"
# Mô hình embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def extract_file_names(text):
    # Sử dụng biểu thức chính quy để tìm các cụm từ kết hợp có dạng "tênfile.đuôifile"
    file_names = re.findall(r'\b(\w+\.\w+)\b', text)
    
    return file_names

def extract_multi_metadata_content(texts, tests):
    extracted_content = ""
    for idx, test in enumerate(tests):
        temp_content = ""
        for x in texts:
            metadata_lower = x.metadata['source'].lower()  # Chuyển nội dung metadata về dạng chữ thường
            if any(term.lower() in metadata_lower for term in test.split()):  # Kiểm tra từng phần của test trong metadata_lower
                temp_content += x.page_content
        if idx == 0:  # Nếu là lần lặp đầu tiên
            extracted_content += f"Dữ liệu của {test}:\n {temp_content}"
        else:
            extracted_content += "\n" + temp_content + "\n"
    return extracted_content

def extract_filename(text):
    # Tìm các từ tiếp theo sau "file" hoặc "tập tin" trong câu
    matches = re.findall(r'\b(?:file|tập\s+tin)\s+(\w+)\b', text.lower())
    return matches

def extract_all_filenames_1(text):
    all_filenames = []
    filenames_1 = extract_file_names(text)
    filenames_2 = extract_filename(text)
    all_filenames.extend(filenames_1)
    all_filenames.extend(filenames_2)
    return all_filenames

def extract_all_filenames(text):
    # Sử dụng biểu thức chính quy để tìm các cụm từ kết hợp có dạng "tênfile.đuôifile"
    file_names_1 = re.findall(r'\b(\w+)\.\w+\b', text)
    
    # Tìm các từ tiếp theo sau "file" hoặc "tập tin" trong câu
    file_names_2 = re.findall(r'\b(?:file|tập\s+tin)\s+(\w+)\b', text.lower())
    
    # Gộp danh sách các tên tệp lại thành một danh sách duy nhất
    all_filenames = file_names_1 + file_names_2
    
    # Tạo một danh sách để lưu trữ các tên tệp duy nhất
    unique_filenames = []
    seen_filenames = set()  # Tập hợp để kiểm tra tên tệp đã xuất hiện
    
    for filename in all_filenames:
        lowercase_filename = filename.lower()
        # Nếu tên tệp chưa xuất hiện trong tập hợp đã thấy, thêm vào danh sách duy nhất và tập hợp đã thấy
        if lowercase_filename not in seen_filenames:
            unique_filenames.append(filename)
            seen_filenames.add(lowercase_filename)
    
    return unique_filenames

def find_matching_files_in_docs(unique_filenames):
    folder_path = "./temp"  # Thay đổi đường dẫn tùy thuộc vào thư mục mà bạn lưu trữ các tệp
    
    matching_files = []
    seen_filenames = set()  # Tập hợp để kiểm tra các tên tệp đã xuất hiện
    
    for filename in unique_filenames:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if filename.lower() in file.lower():
                    if file.lower() not in seen_filenames:
                        matching_files.append(file)
                        seen_filenames.add(file.lower())
                        break  # Nếu tìm thấy một tệp trùng, chuyển sang tên tệp tiếp theo
    return matching_files

def find_matching_files_in_docs_12(text):
    # Thư mục "docs"
    folder_path = "./temp"
    
    # Tạo danh sách để lưu trữ các từ cần tra trong thư mục docs và trong câu truy vấn
    search_terms = []
    search_terms_old = []
    matching_index = []

    
    search_origin = re.findall(r'\b\w+\.\w+\b|\b\w+\b', text)

# Tạo danh sách để lưu trữ các từ cần tìm kiếm, không tách các từ có đuôi file
    search_terms_origin = []
    for word in search_origin:
    # Kiểm tra xem từ có đuôi file không
       if '.' in word:
        search_terms_origin.append(word)
       else:
        # Nếu từ không có đuôi file, tách thành các từ riêng lẻ
        search_terms_origin.extend(re.findall(r'\b\w+\b', word))

    # Tìm tất cả các cụm từ có dạng "tênfile.đuôifile" trong câu và thêm chúng vào danh sách tìm kiếm
    file_names_with_extension = re.findall(r'\b\w+\.\w+\b|\b\w+\b', text.lower())
    file_names_with_extension_old = re.findall(r'\b(\w+\.\w+)\b', text)
    for file_name in search_terms_origin:
        # Kiểm tra xem tên tệp có chứa đuôi file không
        if "." in file_name:
            term_position = search_terms_origin.index(file_name)
            search_terms_old.append(file_name)
    for file_name in file_names_with_extension_old:
        # Kiểm tra xem tên tệp có chứa đuôi file không
        if "." in file_name:
            search_terms_old.append(file_name)
    for file_name in file_names_with_extension:
        # Kiểm tra xem tên tệp có chứa đuôi file không
            search_terms.append(file_name)

    # Tạo biến tạm thời để lưu trữ câu truy vấn sau khi đã loại bỏ các từ tên file.đuôi file
    clean_text_old = text
    clean_text = text.lower()
    for term in search_terms_old:
        clean_text_old = clean_text_old.replace(term, '')
    for term in search_terms:
        clean_text = clean_text.replace(term, '')

    # Tách câu đã xóa các từ tên file.đuôi file thành các từ riêng lẻ và thêm chúng vào danh sách tìm kiếm
    words = re.findall(r'\b\w+\.\w+\b|\b\w+\b', text)
    #search_terms.extend(words)

    words_old = re.findall(r'\b\w+\b', clean_text_old)
    search_terms_old.extend(words_old)

    # Tạo danh sách để duy trì thứ tự của tệp và từ được tìm thấy
     # Tạo tập hợp để lưu trữ các tệp trùng lặp (nếu có)
    matching_files = set()
    matching_files_old = set()
    
    # Tìm các tệp trong thư mục "docs" mà có từ trong danh sách tìm kiếm
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            for term in search_terms:
                if term.lower() in file.lower():
                    term_position = search_terms.index(term)
                    term_value = search_terms_origin[term_position]
                    matching_files.add(file)
                    matching_index.append(term_position)
                    break  # Dừng việc so sánh nếu đã tìm thấy tệp phù hợp
    matching_files_old1 = []
    matching_index.sort()
    for x in matching_index:
        matching_files_old1.append(search_terms_origin[x])
    
    return matching_files,matching_files_old1

def separate_csv_xlsx(files_list):
    list_csv = []
    list_other = []

    for file in files_list:
        if file.endswith('.csv') or file.endswith('.xlsx'):
            list_csv.append(file)
        else:
            list_other.append(file)

    return list_csv, list_other

def convert_xlsx_to_csv(xlsx_file_path, csv_file_path):
    # Read the XLSX file
    df = pd.read_excel(xlsx_file_path)
    df.to_csv(csv_file_path, index=False)


def save_list_CSV(file_list):
    text = ""  # Khởi tạo biến text ở đây để lưu toàn bộ nội dung từ tất cả các tệp
    for x in file_list:
        if x.endswith('.xlsx'):
            old = f"./temp/{x}"
            new = old.replace(".xlsx", ".csv")
            convert_xlsx_to_csv(old, new)
            x = x.replace(".xlsx", ".csv")  # Cập nhật giá trị của x thành tên file CSV mới
        loader1 = CSVLoader(f"./temp/{x}") 
        print(x)
        docs1 = loader1.load()
        text += f"Dữ liệu file {x}:\n"  # Thêm dòng chữ trước nội dung từ mỗi tệp
        for z in docs1:
            text += z.page_content + "\n" # Thêm "\n" để tạo xuống hàng
        
    return text

def extract_query(query,text_alls):
    keyword = find_matching_files_in_docs_12(query)
    list_csv, list_other = separate_csv_xlsx(keyword)
    test_csv = save_list_CSV(list_csv)
    my_set = set(list_other)
    text_document = extract_multi_metadata_content(text_alls,my_set)
    test_all = test_csv + text_document

    return test_all

def chat_gemini(query,text_merge):
    prompt = f"Dựa vào nội dung sau:{text_merge}. Hãy trả lời câu hỏi sau đây: {query}"
    # Set up the model
    generation_config = {
     "temperature": 0.0,
     "top_p": 0.0,
     "top_k": 0,
     "max_output_tokens": 8192,
    }

    safety_settings = [
      {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
   {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
   },
   {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
   },
   {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
   ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

    convo = model.start_chat(history=[])
    convo.send_message(f"{prompt}")
    answer = convo.last.text

    return answer

def extract_content_between_keywords(query, keywords):
    contents = {}
    num_keywords = len(keywords)
    keyword_positions = []
    
    for i in range(num_keywords):
        keyword = keywords[i]
        # Xác định vị trí của từ khóa trong câu truy vấn
        keyword_position = query.find(keyword)
        keyword_positions.append(keyword_position)
        
        # Nếu từ khóa không tồn tại trong câu truy vấn, bỏ qua
        if keyword_position == -1:
            continue
        
        # Tìm vị trí của từ khóa tiếp theo sau từ khóa hiện tại
        next_keyword_position = len(query)
        for j in range(i + 1, num_keywords):
            next_keyword = keywords[j]
            next_keyword_position = query.find(next_keyword)
            if next_keyword_position != -1:
                break
        
        # Trích xuất nội dung trước từ khóa đầu tiên
        if i == 0:
            content_before = query[:keyword_position].strip()
        else:
            content_before = query[keyword_positions[i-1] + len(keywords[i-1]):keyword_position].strip()
        
        # Trích xuất nội dung sau từ khóa cuối cùng
        if i == num_keywords - 1:
            content_after = query[keyword_position + len(keyword):].strip()
        else:
            content_after = query[keyword_position + len(keyword):next_keyword_position].strip()
        
        # Ghép từ khóa với nội dung trước và sau để tạo thành câu hoàn chỉnh
        content = f"{content_before} {keyword} {content_after}"
        
        # Lưu câu hoàn chỉnh vào từ điển
        contents[keyword] = content
    
    return contents

def merge_files(file_set, file_list):
    """Hàm này ghép lại các tên file dựa trên điều kiện đã cho."""
    merged_files = {}
    
    # Ghép lại các tên file từ file_list
    for file_name in file_list:
        name = file_name.split('.')[0]
        for f in file_set:
            if name in f:
                merged_files[name] = f
                break
        
    return merged_files

def replace_keys_with_values(original_dict, replacement_dict):
    """
    Thay thế các key trong original_dict bằng các giá trị tương ứng từ replacement_dict.
    
    Tham số:
        - original_dict: Từ điển gốc cần thay đổi.
        - replacement_dict: Từ điển chứa các cặp key-value sẽ được sử dụng để thay thế key trong original_dict.
    
    Trả về:
        - new_dict: Từ điển mới sau khi thực hiện thay đổi.
    """
    new_dict = {}
    for key, value in original_dict.items():
        if key in replacement_dict:
            new_key = replacement_dict[key]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

def aws1_csv(new_dict_csv):
 text = ""
 query_all = ""
 for key, value in new_dict_csv.items():
    print(key,value)
    query = value
    query_all += value
    keyword = []
    keyword.append(key)
    print(keyword)
    test = save_list_CSV(keyword)
    text += test
 return text,query_all


def aws1(new_dict,text_alls):
 text = ""
 query_all = ""
 for key, value in new_dict.items():
    query = value
    query_all += value
    keyword,keyword2=find_matching_files_in_docs_12(query)
    print(value)
    print(keyword)
    data= extract_multi_metadata_content(text_alls,keyword)
    #Phân chia dữ liệu này lại và rerank
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2200, chunk_overlap=400)
    texts_data = text_splitter.split_text(data)
    persist_directory = f'{key}'
    vectordb = Chroma.from_texts(texts_data,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
    k_1 = round(len(texts_data))
    
    retriever = vectordb.as_retriever(search_kwargs={f"k":k_1})
    llm = Cohere(temperature=0)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.get_relevant_documents(
    f"{query}"
    )
    text += "Dữ liệu file" + f"{key}"
    i =0 
    for x in compressed_docs:
      text += x.page_content
      i= i +1
    
 return text,query_all
    
from groq import Groq

def get_chat_completion(prompt_query):
 try:
  chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        {
            "role": "system",
            "content": "Bạn là một trợ lý trung thưc, trả lời dựa trên nội dung tài liệu được cung cấp. Chỉ trả lời liên quan đến câu hỏi một cách đầy đủ chính xác, không bỏ sót thông tin."
        },
        {
            "role": "user",
            "content": f"{prompt_query}",
        }
    ],

    # The language model which will generate the completion.
    model="llama3-70b-8192",
    temperature=0.0,
    # The maximum number of tokens to generate. Requests can use up to
    # 2048 tokens shared between prompt and completion.
    max_tokens=9000,

    # Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    #top_p=1,

    # A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    stop=None,

    # If set, partial message deltas will be sent.
    stream=False,
  )
  return chat_completion.choices[0].message.content
 except Exception as error:
    # Handle the RateLimitError here
    #print("Rate limit reached. Please try again later.")
    #print("Error message:", error.message)
    return False
# Print the completion returned by the LLM.

def initialize_generative_model(prompt):
    # Set up the model generation configuration
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0,
        "max_output_tokens": 8192,
    }

    # Define safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    # Initialize the generative model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

    # Start a conversation
    convo = model.start_chat(history=[])

    # Send the prompt
    convo.send_message(prompt)

    # Return the last response
    return convo.last.text

# Example usage:

def question_answer(question):
    completion = get_chat_completion(question)
    if completion:
        return completion
    else:
        answer = initialize_generative_model(question)
        return answer

def aws1_all(new_dict,text_alls):
 answer = ""
 for key, value in new_dict.items():
    query = value
    keyword,keyword2=find_matching_files_in_docs_12(query)
    print(value)
    print(keyword)
    data= extract_multi_metadata_content(text_alls,keyword)
    #Phân chia dữ liệu này lại và rerank
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2200, chunk_overlap=400)
    texts_data = text_splitter.split_text(data)
    persist_directory = f'{key}'
    vectordb = Chroma.from_texts(texts_data,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
    k_1 = round(len(texts_data))
    
    retriever = vectordb.as_retriever(search_kwargs={f"k":k_1})
    llm = Cohere(temperature=0)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.get_relevant_documents(
    f"{query}"
    )
    text = ""
    text += "Dữ liệu file" + f"{key}"
    i =0 
    for x in compressed_docs:
      text += x.page_content
      i= i +1
    
    prompt_document = f"Dựa vào nội dung sau:{text}. Hãy trả lời câu hỏi sau đây: {query}"
    answer_for = question_answer(prompt_document)
    answer += answer_for + "\n"

    
 return answer

def question_answer_all_query_v1(query,text_alls):
    keyword1,key_words_old=find_matching_files_in_docs_12(query)
    list_keywords2 = list(key_words_old)
    contents1 = extract_content_between_keywords(query, list_keywords2)
    merged_result = merge_files(keyword1,list_keywords2)
    original_dict = contents1
    # Từ điển replacement
    replacement_dict = merged_result
    new_dict = replace_keys_with_values(original_dict, replacement_dict)

    files_to_remove = [filename for filename in new_dict.keys() if filename.endswith('.xlsx') or filename.endswith('.csv')]
    removed_files = {}

    for filename in files_to_remove:
       removed_files[filename] = new_dict[filename]

# Xóa các tệp khỏi new_dict
    for filename in files_to_remove:
      new_dict.pop(filename)
    test_csv = ""
    text_csv,query_csv = aws1_csv(removed_files)
    prompt_csv = ""
    answer_csv = ""
    if test_csv:
        prompt_csv = f"Dựa vào nội dung sau: {text_csv}. Hãy trả lời câu hỏi sau đây: {query_csv}.Bằng tiếng Việt"
        answer_csv = question_answer(prompt_csv)

    answer_document = aws1_all(new_dict,text_alls)

    

    answer_all = answer_document + answer_csv
    
    return answer_all

def check_both_empty(matching_files, matching_files_old):
    """
    Kiểm tra nếu cả hai biến đều rỗng.

    Args:
    - matching_files: Danh sách các tệp phù hợp mới.
    - matching_files_old: Danh sách các tệp phù hợp cũ.

    Returns:
    - True nếu cả hai biến đều rỗng, False nếu không.
    """
    return not matching_files and not matching_files_old

def question_answer_all_query(query):
    keyword1,key_words_old=find_matching_files_in_docs_12(query)
    list_keywords2 = list(key_words_old)
    contents1 = extract_content_between_keywords(query, list_keywords2)
    merged_result = merge_files(keyword1,list_keywords2)
    original_dict = contents1
    # Từ điển replacement
    replacement_dict = merged_result
    new_dict = replace_keys_with_values(original_dict, replacement_dict)

    files_to_remove = [filename for filename in new_dict.keys() if filename.endswith('.xlsx') or filename.endswith('.csv')]
    removed_files = {}

    for filename in files_to_remove:
       removed_files[filename] = new_dict[filename]

# Xóa các tệp khỏi new_dict
    for filename in files_to_remove:
      new_dict.pop(filename)

    text_document, query_document = aws1(new_dict)
    test_csv = ""
    text_csv,query_csv = aws1_csv(removed_files)
    if test_csv:
       prompt_csv = f"Dựa vào nội dung sau: {text_csv}. Hãy trả lời câu hỏi sau đây: {query_csv}.Bằng tiếng Việt"
    prompt_document = f"Dựa vào nội dung sau: {text_document}. Hãy trả lời câu hỏi sau đây: {query_document}. Bằng tiếng Việt"
    
    

    answer_document = question_answer(prompt_document)

    answer_csv = question_answer(prompt_csv)

    answer_all = answer_document + answer_csv
    
    return answer_all

def extract_data():
 documents = []
# Load dữ liệu các file từ thư mục docs
 for file in os.listdir("./temp"):
    if file.endswith(".pdf"):
        pdf_path = "./temp/" + file
        loader = UnstructuredPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./temp/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        txt_path = "./temp/" + file
        loader = TextLoader(txt_path,encoding="utf8")
        documents.extend(loader.load())
    elif file.endswith('.pptx'):
        ppt_path = "./temp/" + file
        loader = UnstructuredPowerPointLoader(ppt_path)
        documents.extend(loader.load())
    elif file.endswith('.csv'):
        csv_path = "./temp/" + file
        loader = UnstructuredCSVLoader(csv_path)
        documents.extend(loader.load())
    elif file.endswith('.xlsx'):
        excel_path = "./temp/" + file
        loader = UnstructuredExcelLoader(excel_path)
        documents.extend(loader.load())
    elif file.endswith('.xml'):
        xml_path = "./temp/" + file
        loader = UnstructuredXMLLoader(xml_path)
        documents.extend(loader.load())
    elif file.endswith('.html'):
        html_path = "./temp/" + file
        loader = UnstructuredHTMLLoader(html_path)
        documents.extend(loader.load())
    elif file.endswith('.json'):
        json_path = "./temp/" + file
        loader = JSONLoader(json_path)
        documents.extend(loader.load())
    elif file.endswith('.md'):
        json_path = "./temp/" + file
        loader = UnstructuredMarkdownLoader(json_path)
        documents.extend(loader.load())  
 #Phân chia dữ liệu
 text_splitter = CharacterTextSplitter(chunk_size=2200, chunk_overlap=1500)
 texts= text_splitter.split_documents(documents)
 text_all = texts

 return text_all





