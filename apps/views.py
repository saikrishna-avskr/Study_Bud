from django.shortcuts import render
from django.http import HttpResponse
import os
from google.cloud import vision
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
from dotenv import load_dotenv
import google.generativeai as genai2
from tabulate import tabulate
from datetime import datetime
import numpy as np  
import cv2
import tempfile
import json
import html2text

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
        
        # Enhanced prompt for better formatting
        prompt = f"""
        {problem_statement}
        
        Please provide:
        1. A complete, working program in {language}
        2. A detailed explanation of the code in {explanguage}
        
        Format your response as follows:
        - Start with a brief description
        - Put the code in a proper code block with syntax highlighting for {language} using this format:
          ```{language.lower()}
          // your code here
          ```
        - Follow with a detailed explanation using proper markdown formatting with headers, lists, and inline code where appropriate
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        )
        
        # Configure markdown with syntax highlighting support
        markdown_extensions = ['codehilite', 'fenced_code', 'tables', 'toc']
        formatted_response = markdown.markdown(
            response.text, 
            extensions=markdown_extensions,
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': True
                }
            }
        )
        
        return render(request, "code_generator.html", {"response": formatted_response})
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

credentials_info = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("UNIVERSE_DOMAIN")
}

### Uncomment the following code if you are deploying on locally
# credentials_path = os.path.join(tempfile.gettempdir(), "credentials.json")
credentials_path = "/tmp/credentials.json"
with open(credentials_path, "w") as f:
    json.dump(credentials_info, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

credentials = service_account.Credentials.from_service_account_info(credentials_info)

def generate_image(request): 
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        init_vertex_ai(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"), credentials_path=credentials_path)
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
            request.session["evaluations"] = []  # Initialize evaluations storage
        else:
            questions = request.session.get("questions", [])
        
        total_questions = len(questions)
        current_question_index = int(request.POST.get("current_question_index", 0))
        
        # Store current answer and evaluation before navigation
        user_answer = request.POST.get("user_answer", "")
        evaluation_response = "Evaluation not available"
        
        if user_answer and questions and current_question_index < len(questions):
            current_question = questions[current_question_index]
            if user_answer.strip():  # Only evaluate non-empty answers
                response = client.models.generate_content(
                    model='gemini-2.0-flash', 
                    contents=f"Evaluate the following answer: {user_answer} for the question: {current_question}. Grade the answer. Explain the answer if it is wrong. Give ways to improve the answer if needed.",
                )
                evaluation_response = response.text
                
                # Store the evaluation data
                evaluations = request.session.get("evaluations", [])
                
                # Update or append evaluation for current question
                evaluation_data = {
                    "question_index": current_question_index,
                    "question": current_question,
                    "answer": user_answer,
                    "evaluation": evaluation_response,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check if evaluation for this question already exists
                existing_index = None
                for i, eval_item in enumerate(evaluations):
                    if eval_item["question_index"] == current_question_index:
                        existing_index = i
                        break
                
                if existing_index is not None:
                    evaluations[existing_index] = evaluation_data
                else:
                    evaluations.append(evaluation_data)
                
                request.session["evaluations"] = evaluations
        
        # Handle navigation
        if "next" in request.POST and current_question_index < total_questions - 1:
            current_question_index += 1
        elif "prev" in request.POST and current_question_index > 0:
            current_question_index -= 1
        
        # Get current question for display
        current_question = questions[current_question_index] if questions else ""
        
        # Get stored answer for current question
        evaluations = request.session.get("evaluations", [])
        stored_answer = ""
        stored_evaluation = "Evaluation not available"
        
        for eval_item in evaluations:
            if eval_item["question_index"] == current_question_index:
                stored_answer = eval_item["answer"]
                stored_evaluation = eval_item["evaluation"]
                break
        
        # Check if test is complete (all questions have been answered)
        answered_questions = len([e for e in evaluations if e["answer"].strip()])
        is_complete = answered_questions == total_questions
        
        context = {
            "current_question": current_question,
            "current_question_index": current_question_index,
            "total_questions": total_questions,
            "evaluation_response": markdown.markdown(stored_evaluation),
            "user_answer": stored_answer,
            "intype": intype,
            "is_complete": is_complete,
            "answered_questions": answered_questions,
        }
        
        return render(request, "evaluate_assignment.html", context)    
    return render(request, "upload_assignment.html")

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

    Please format your response using proper markdown with:
    - Clear headings (use ##, ###, ####)
    - Organized sections with day-by-day breakdown
    - Bullet points and numbered lists where appropriate
    - Bold text for important concepts
    - Italic text for emphasis
    - A well-structured table for the daily schedule at the end
    
    Structure the response as:
    1. ## Introduction and Overview
    2. ## Detailed Day-by-Day Study Plan
    3. ## Important Tips and Considerations
    4. ## Daily Schedule Table
    
    For the Daily Schedule Table, use this exact format:
    
    | Time Slot | Day 1 | Day 2 | Day 3 | Day 4 |
    |-----------|-------|-------|-------|-------|
    | 8:00 AM - 10:00 AM | Topic 1 | Topic 2 | Topic 3 | Topic 4 |
    | 10:00 AM - 12:00 PM | Topic 1 | Topic 2 | Topic 3 | Topic 4 |
    
    Make sure the table is properly formatted with clear column separators (|) and includes all study sessions.
    """
    try:
        response = model.generate_content(prompt) 
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def extract_table(study_plan_text):
    # Look for various table patterns
    table_patterns = [
        "Time Slot",
        "Day\t",
        "| Day |",
        "Day\tMonday",
        "## 4. Daily Schedule Table"
    ]
    
    table_start = -1
    for pattern in table_patterns:
        table_start = study_plan_text.find(pattern)
        if table_start != -1:
            break
    
    if table_start == -1:
        return None

    # Extract table section
    table_section = study_plan_text[table_start:]
    
    # Find the end of table (next major section or end of text)
    table_end = len(table_section)
    for end_marker in ["\n## ", "\n# ", "\n---", "\nConclusion"]:
        end_pos = table_section.find(end_marker, 100)  # Look after first 100 chars
        if end_pos != -1:
            table_end = min(table_end, end_pos)
    
    table_text = table_section[:table_end].strip()
    lines = table_text.split('\n')
    
    # Filter out empty lines and non-table content
    table_lines = []
    for line in lines:
        line = line.strip()
        if line and ('|' in line or 'Time Slot' in line or any(day in line for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'])):
            table_lines.append(line)
    
    if len(table_lines) < 2:
        return None
    
    # Process table data
    table_data = []
    header = None
    
    for line in table_lines:
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                if header is None:
                    header = cells
                else:
                    table_data.append(cells)
        elif 'Time Slot' in line and header is None:
            # Handle tab-separated or space-separated header
            header = ['Time Slot', 'Monday (2025-08-26)', 'Tuesday (2025-08-27)', 'Wednesday (2025-08-28)', 'Thursday (2025-08-29)']
        elif any(time in line for time in ['8:00 AM', '9:30 AM', '11:00 AM', '1:00 PM', '2:30 PM', '4:00 PM']):
            # Handle tab or space separated data rows
            cells = line.split('\t') if '\t' in line else line.split()
            if len(cells) >= 2:
                table_data.append(cells)
    
    if header and table_data:
        return tabulate(table_data, headers=header, tablefmt="html", maxcolwidths=[15, 20, 20, 20, 20])
    else:
        return None  
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

        # Enhanced markdown rendering with extensions
        markdown_extensions = ['tables', 'toc', 'fenced_code']
        formatted_study_plan = markdown.markdown(
            study_plan_text, 
            extensions=markdown_extensions
        )

        return render(request, 'study_plan.html', {
            'study_plan_text': formatted_study_plan, 
            'table': table
        })

    return render(request, 'study_plan.html')  

client1 = vision.ImageAnnotatorClient(credentials=credentials)

def annotate_image(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client1.label_detection(image=image)
    labels = response.label_annotations

    response = client1.object_localization(image=image)
    objects = response.localized_object_annotations

    return labels, objects

def draw_bounding_boxes(image_path, objects):
    image = cv2.imread(image_path)

    for obj in objects:
        vertices = [(int(vertex.x * image.shape[1]), int(vertex.y * image.shape[0])) for vertex in obj.bounding_poly.normalized_vertices]
        cv2.polylines(image, [np.array(vertices)], isClosed=True, color=(0, 255, 0), thickness=2)
        object_name = obj.name
        score = obj.score
        label = f"{object_name} ({score*100:.2f}%)"
        top_left = (vertices[0][0], vertices[0][1] - 10)
        cv2.putText(image, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, buffer = cv2.imencode('.png', image)
    image_bytes = buffer.tobytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    return encoded_image


def annotate_image_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_path = BASE_DIR / "temp_image.jpg"
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        labels, objects = annotate_image(image_path)
        base64_image = draw_bounding_boxes(image_path, objects)
        prompt = f"""
        Use the following {labels} and give me a description about the entity present.
        Generate a list of 10 questions based on the same along with answers.
        """
        response = model.generate_content(prompt) 
        return render(request, 'annotated_image.html', {
            'labels': labels,
            'objects': objects,
            'base64_image': base64_image,
            'prompttext': markdown.markdown(response.text)
            
        })

    return render(request, 'upload_image.html')
def pomodoro(request):
    return render(request, 'pomodoro.html')