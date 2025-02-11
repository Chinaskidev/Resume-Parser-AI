from fastapi import FastAPI, UploadFile, File
import PyPDF2
import docx2txt
import re
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lista  de habilidades blandas y fuertes que buscan las empresas.
SKILLS_LIST = {"python", "java", "javascript", "sql", "machine learning", "data analysis",
               "react", "aws", "licenciado en Administracion de Empresas", "Economista",
               "Auditor", "Cloud Computing", "Inteligencia Artificial", "Gestión de personas", 
               "Diseño UX", "Desarrollo de aplicaciones móviles", "Producción de video",
               "Liderazgo de ventas","raducción", "Producción de audio", "Procesamiento del lenguaje natural",
               "Trabajar en equipo", "Resolver conﬂictos y problemas", "Capacidad de tomar decisiones",
               "Adaptación al cambio", "Capacidad de comunicar eficazmente", "Proactividad", "Empatía",
               "Creatividad", "Tolerancia a la presión", "Orientación a resultados"            
               "Compromiso", "Capacidad de Aprendizaje", "Innovación", "Impacto/Influencia", "Resolución sostenible de conflictos"               
               }

def extract_text(file: UploadFile) -> str:
    """Extrae texto de un archivo PDF o DOCX."""
    text = ""
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif file.filename.endswith(".docx"):
        text = docx2txt.process(file.file)
    return text.lower()  # text.lower convierte en minusculas el texto asi no hay fallos cuando escriben...


def extract_skills(text: str) -> list:
    """Extrae habilidades del texto buscando coincidencias en la lista predefinida."""
    found_skills = [skill for skill in SKILLS_LIST if skill in text]
    return found_skills


def extract_experience(text: str) -> list:
    """Extrae experiencia en años usando expresiones regulares."""
    experience = re.findall(r"(\d+)\s*(?:años|years)", text)
    return experience if experience else []


def match_resume_to_job(resume_text: str, job_desc: str) -> float:
    """Calcula la similitud entre el resume y la descripción del trabajo."""
    embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(score, 2)


@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile = File(...), job_desc: str = ""):
    """API para analizar un resume en base a una descripción de trabajo."""
    resume_text = extract_text(file)
    skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)
    match_score = match_resume_to_job(resume_text, job_desc)

    
    """El valor 0.5 (50%) es un umbral que decide si un
    candidato es seleccionado o no en base al puntaje de coincidencia (match_score). Podemos cambiar
    dependiendo del valor que quieran darle. """
    result = {
        "file_name": file.filename,
        "match_score": match_score,
        "skills": skills,
        "experience": experience,
        "decision": "Selected" if match_score > 0.5 else "No fue seleccionado",
        "reason": "Good match" if match_score > 0.5 else "Falta de experiencia o habilidades relevantes"
    }
    return result
