from django.shortcuts import render
import os
import markdown
from pathlib import Path
import typing
from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
from google.oauth2 import service_account
import io
import base64
from PIL import Image
from google import genai
import PyPDF2
import markdown
from dotenv import load_dotenv
import google.generativeai as genai2
from tabulate import tabulate
from datetime import datetime


load_dotenv()
genai2.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai2.GenerativeModel('gemini-2.0-flash')


BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):
    return render(request,'index.html')

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_code(request):
    if request.method=="POST":
        problem_statement = request.POST.get("problem_statement")
        language = request.POST.get("language")     
        explanguage = request.POST.get("explanguage")   
        if explanguage is None:
            explanguage='english'
        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=problem_statement+". Generate program in "+language+". Explain the code in "+explanguage+".",
        )
        return render(request, "code_generator.html", {"response": markdown.markdown(response.text)})
    else:
        return render(request,'code_generator.html')

def init_vertex_ai(project: str, location: str, credentials_path: str) -> None:
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    vertexai.init(project=project, location=location, credentials=credentials)

def generate_images_from_prompt(
    prompt: str, 
    number_of_images: int = 1, 
    aspect_ratio: str = "1:1", 
    negative_prompt: str = "", 
    safety_filter_level: str = "", 
    add_watermark: bool = True
) -> typing.List[typing.Any]:
    generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")    
    images = generation_model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt,
        person_generation="",
        safety_filter_level=safety_filter_level,
        add_watermark=add_watermark
    )    
    return images

def image_to_base64(image: typing.Any) -> str:
    buffered = io.BytesIO()
    pil_image = Image.open(io.BytesIO(image._image_bytes))  
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_image(request): 
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        credentials_path = BASE_DIR / "credentials.json"
        init_vertex_ai(project="gen-lang-client-0978853110", location="us-central1", credentials_path=credentials_path)
        images = generate_images_from_prompt(prompt=prompt)
        image_base64 = image_to_base64(images[0])
        return render(request, "image_generation.html", {"image_base64": image_base64})
    else:
        return render(request, "image_generation.html") 



def evaluate_assignment(request):
    if request.method == "POST":
        intype = request.POST.get("intype")
        if "questions" not in request.session or intype:
            questions = []
            if intype == "pdf":
                pdf_file = request.FILES.get("pdf_file")
                if pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        questions.extend(page.extract_text().split("\n"))
            elif intype == "text":
                text_input = request.POST.get("text_input")
                questions = text_input.split("\n")
            
            questions = [q.strip() for q in questions if q.strip()]
            request.session["questions"] = questions
        else:
            questions = request.session.get("questions", [])
        total_questions = len(questions)
        current_question_index = int(request.POST.get("current_question_index", 0))
        if "next" in request.POST and current_question_index < total_questions - 1:
            current_question_index += 1
        elif "prev" in request.POST and current_question_index > 0:
            current_question_index -= 1
        current_question = questions[current_question_index] if questions else ""
        user_answer = request.POST.get("user_answer", "")
        evaluation_response = "Evaluation not available"
        if user_answer and current_question:
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=f"Evaluate the following answer: {user_answer} for the question: {current_question}.Grade the answer. Explain the answer if it is wrong. Give ways to improve the answer if needed.",
            )
            evaluation_response = response.text
        context = {
            "current_question": current_question,
            "current_question_index": current_question_index,
            "total_questions": total_questions,
            "evaluation_response": markdown.markdown(evaluation_response),
            "user_answer": user_answer,        }
        
        return render(request, "evaluate_assignment.html", context)    
    return render(request, "upload_assignment.html")


from datetime import datetime
import markdown
from tabulate import tabulate  

def get_current_date():
    return datetime.now().date()

def generate_study_plan(user_inputs):
    prompt = f"""
    Generate a detailed study plan based on the following information:

    Subject: {user_inputs['subject']}
    Today is {get_current_date()}
    Exam Date: {user_inputs['exam_date']}
    Available Study Time per Day: {user_inputs['study_time']}
    Specific Topics to Focus On: {user_inputs['topics']}
    Preferred Study Methods: {user_inputs['methods']}
    Any Other Important Notes: {user_inputs['notes']}

    Please provide the study plan in two parts:
    1. A detailed study plan with a day-by-day breakdown.
    2. A table summarizing the daily study schedule.
    """
    try:
        response = model.generate_content(prompt) 
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def extract_table(study_plan_text):
    table_start = study_plan_text.find("Day")
    if table_start == -1:
        return "Table not found."

    table_string = study_plan_text[table_start:] 
    lines = table_string.strip().split('\n')
    table_data = []
    header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
    for line in lines[1:]:
        row = [cell.strip() for cell in line.split('|') if cell.strip()]
        if row:
            table_data.append(row)
    return tabulate(table_data, headers=header, tablefmt="html")  
def study_plan_view(request):
    if request.method == 'POST':
        user_inputs = {
            'subject': request.POST.get('subject'),
            'exam_date': request.POST.get('exam_date'),
            'study_time': request.POST.get('study_time'),
            'topics': request.POST.get('topics'),
            'methods': request.POST.get('methods'),
            'notes': request.POST.get('notes'),
        }
        study_plan_text = generate_study_plan(user_inputs)
        table = None
        try:
            table = extract_table(study_plan_text)
        except Exception as e:
            error_message = f"Could not generate table: {e}"
            return render(request, 'study_plan.html', {'study_plan_text': study_plan_text, 'error': error_message})

        return render(request, 'study_plan.html', {'study_plan_text': markdown.markdown(study_plan_text), 'table':table})

    return render(request, 'study_plan.html')  

