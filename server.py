from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
JWT_EXPIRATION = int(os.environ.get('JWT_EXPIRATION_HOURS', '24'))

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Models ====================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime

class InterviewSettings(BaseModel):
    competencies: List[str] = Field(default=["Teamwork", "Leadership", "Conflict Management", "Ownership"])
    num_questions: int = Field(default=5, ge=3, le=10)
    difficulty: str = Field(default="medium")  # easy, medium, hard

class InterviewStart(BaseModel):
    settings: InterviewSettings

class InterviewResponse(BaseModel):
    id: str
    user_id: str
    status: str  # active, completed
    settings: InterviewSettings
    current_question_index: int
    total_score: int
    created_at: datetime
    completed_at: Optional[datetime] = None

class QuestionResponse(BaseModel):
    id: str
    interview_id: str
    question_text: str
    competency: str
    difficulty: str
    order: int

class STARScore(BaseModel):
    situation: int = Field(ge=0, le=5)
    task: int = Field(ge=0, le=5)
    action: int = Field(ge=0, le=5)
    result: int = Field(ge=0, le=5)
    total: int = Field(ge=0, le=20)

class Evaluation(BaseModel):
    scores: STARScore
    strengths: List[str]
    improvements: List[str]
    follow_up_question: Optional[str] = None

class AnswerSubmit(BaseModel):
    answer_text: str

class ResponseWithEvaluation(BaseModel):
    id: str
    question_id: str
    answer_text: str
    evaluation: Evaluation
    timestamp: datetime
    next_question: Optional[QuestionResponse] = None
    interview_completed: bool = False

class InterviewSummary(BaseModel):
    interview_id: str
    total_questions: int
    overall_score: int
    max_score: int
    average_score: float
    top_strengths: List[str]
    top_improvements: List[str]
    competency_breakdown: Dict[str, Dict[str, Any]]
    readiness_level: str
    personalized_advice: List[str]

# ==================== Helper Functions ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION)
    payload = {
        "user_id": user_id,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    user_id = decode_jwt_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    return user

# ==================== AI Service ====================

class AIInterviewService:
    def __init__(self):
        self.api_key = EMERGENT_LLM_KEY
    
    async def generate_first_question(self, settings: InterviewSettings, interview_id: str) -> str:
        """Generate the first question for the interview"""
        competency = settings.competencies[0] if settings.competencies else "Teamwork"
        
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"interview_{interview_id}_first_q",
            system_message="You are a professional behavioural interview expert. Generate realistic, scenario-based interview questions."
        ).with_model("openai", "gpt-4o")
        
        prompt = f"""Generate ONE behavioural interview question focused on: {competency}

Difficulty level: {settings.difficulty}

The question should:
- Be scenario-based ("Tell me about a time when...")
- Be realistic and job-relevant
- Target {competency.lower()} competency
- Be appropriate for {settings.difficulty} difficulty

Provide ONLY the question text, nothing else."""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        return response.strip()
    
    async def generate_next_question(self, interview_id: str, settings: InterviewSettings, 
                                    current_index: int, conversation_history: List[Dict]) -> str:
        """Generate the next question based on conversation history"""
        if current_index >= len(settings.competencies):
            # Cycle through competencies
            competency = settings.competencies[current_index % len(settings.competencies)]
        else:
            competency = settings.competencies[current_index]
        
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"interview_{interview_id}_q{current_index}",
            system_message="You are a professional behavioural interview expert."
        ).with_model("openai", "gpt-4o")
        
        # Build context from previous answers
        context = "Previous questions and performance:\n"
        for item in conversation_history[-3:]:  # Last 3 for context
            context += f"- Q: {item['question']}\n"
            context += f"  Score: {item['score']}/20\n"
        
        prompt = f"""{context}

Generate the NEXT behavioural interview question focused on: {competency}

Difficulty level: {settings.difficulty}

The question should:
- Be different from previous questions
- Target {competency.lower()} competency
- Be scenario-based
- Consider the candidate's performance (adapt if needed)

Provide ONLY the question text, nothing else."""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        return response.strip()
    
    async def evaluate_answer(self, question: str, answer: str, competency: str, 
                            interview_id: str, question_order: int) -> Evaluation:
        """Evaluate answer using STAR framework"""
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"interview_{interview_id}_eval_{question_order}",
            system_message="""You are an expert behavioural interview evaluator. 
Evaluate answers using the STAR method (Situation, Task, Action, Result).
Provide scores from 0-5 for each component, specific strengths, and actionable improvements."""
        ).with_model("openai", "gpt-4o")
        
        prompt = f"""Question: {question}
Competency being tested: {competency}

Candidate's Answer: {answer}

Evaluate this answer using the STAR framework:

1. Situation (0-5): Is the context clear and specific?
2. Task (0-5): Is the responsibility clearly defined?
3. Action (0-5): Are actions detailed and personally owned?
4. Result (0-5): Are outcomes clear, measurable, or reflective?

Provide your evaluation in this EXACT JSON format:
{{
  "situation_score": <number 0-5>,
  "task_score": <number 0-5>,
  "action_score": <number 0-5>,
  "result_score": <number 0-5>,
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "follow_up": "A focused follow-up question targeting the weakest area"
}}

Respond ONLY with valid JSON, no other text."""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        
        try:
            # Parse JSON response
            eval_data = json.loads(response.strip())
            
            total_score = (eval_data["situation_score"] + eval_data["task_score"] + 
                          eval_data["action_score"] + eval_data["result_score"])
            
            return Evaluation(
                scores=STARScore(
                    situation=eval_data["situation_score"],
                    task=eval_data["task_score"],
                    action=eval_data["action_score"],
                    result=eval_data["result_score"],
                    total=total_score
                ),
                strengths=eval_data["strengths"][:3],
                improvements=eval_data["improvements"][:3],
                follow_up_question=eval_data.get("follow_up")
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing evaluation: {e}")
            # Fallback evaluation
            return Evaluation(
                scores=STARScore(situation=3, task=3, action=3, result=3, total=12),
                strengths=["Answer provided"],
                improvements=["Could provide more specific details", "Consider using STAR framework"],
                follow_up_question="Can you elaborate on the specific actions you took?"
            )

ai_service = AIInterviewService()

# ==================== Routes ====================

@api_router.get("/")
async def root():
    return {"message": "Behavioural Interview Simulator API"}

# Auth Routes
@api_router.post("/auth/register", response_model=dict)
async def register(user_data: UserRegister):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    hashed_pw = hash_password(user_data.password)
    
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": hashed_pw,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    token = create_jwt_token(user_id)
    
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name
        }
    }

@api_router.post("/auth/login", response_model=dict)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(user["id"])
    
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"]
        }
    }

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=datetime.fromisoformat(current_user["created_at"])
    )

# Interview Routes
@api_router.post("/interviews/start", response_model=dict)
async def start_interview(interview_data: InterviewStart, current_user: dict = Depends(get_current_user)):
    interview_id = str(uuid.uuid4())
    
    # Generate first question
    first_question = await ai_service.generate_first_question(interview_data.settings, interview_id)
    question_id = str(uuid.uuid4())
    
    # Create interview
    interview_doc = {
        "id": interview_id,
        "user_id": current_user["id"],
        "status": "active",
        "settings": interview_data.settings.model_dump(),
        "current_question_index": 0,
        "total_score": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None
    }
    
    # Create first question
    question_doc = {
        "id": question_id,
        "interview_id": interview_id,
        "question_text": first_question,
        "competency": interview_data.settings.competencies[0],
        "difficulty": interview_data.settings.difficulty,
        "order": 0,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.interviews.insert_one(interview_doc)
    await db.questions.insert_one(question_doc)
    
    return {
        "interview_id": interview_id,
        "first_question": {
            "id": question_id,
            "question_text": first_question,
            "competency": interview_data.settings.competencies[0],
            "order": 0
        },
        "total_questions": interview_data.settings.num_questions
    }

@api_router.post("/interviews/{interview_id}/respond", response_model=ResponseWithEvaluation)
async def submit_answer(interview_id: str, answer_data: AnswerSubmit, 
                       current_user: dict = Depends(get_current_user)):
    # Get interview
    interview = await db.interviews.find_one({"id": interview_id, "user_id": current_user["id"]})
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    if interview["status"] != "active":
        raise HTTPException(status_code=400, detail="Interview is not active")
    
    # Get current question
    current_order = interview["current_question_index"]
    question = await db.questions.find_one({"interview_id": interview_id, "order": current_order})
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Evaluate answer
    evaluation = await ai_service.evaluate_answer(
        question["question_text"],
        answer_data.answer_text,
        question["competency"],
        interview_id,
        current_order
    )
    
    # Save response
    response_id = str(uuid.uuid4())
    response_doc = {
        "id": response_id,
        "question_id": question["id"],
        "interview_id": interview_id,
        "answer_text": answer_data.answer_text,
        "evaluation": evaluation.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.responses.insert_one(response_doc)
    
    # Update interview score
    new_total_score = interview["total_score"] + evaluation.scores.total
    next_question_index = current_order + 1
    
    settings = InterviewSettings(**interview["settings"])
    
    # Check if interview is complete
    interview_completed = next_question_index >= settings.num_questions
    
    next_question = None
    if not interview_completed:
        # Generate next question
        conversation_history = []
        async for resp in db.responses.find({"interview_id": interview_id}):
            q = await db.questions.find_one({"id": resp["question_id"]})
            conversation_history.append({
                "question": q["question_text"],
                "score": resp["evaluation"]["scores"]["total"]
            })
        
        next_q_text = await ai_service.generate_next_question(
            interview_id, settings, next_question_index, conversation_history
        )
        
        next_q_id = str(uuid.uuid4())
        competency_index = next_question_index % len(settings.competencies)
        
        next_q_doc = {
            "id": next_q_id,
            "interview_id": interview_id,
            "question_text": next_q_text,
            "competency": settings.competencies[competency_index],
            "difficulty": settings.difficulty,
            "order": next_question_index,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.questions.insert_one(next_q_doc)
        
        next_question = QuestionResponse(
            id=next_q_id,
            interview_id=interview_id,
            question_text=next_q_text,
            competency=settings.competencies[competency_index],
            difficulty=settings.difficulty,
            order=next_question_index
        )
    
    # Update interview
    update_data = {
        "total_score": new_total_score,
        "current_question_index": next_question_index
    }
    
    if interview_completed:
        update_data["status"] = "completed"
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.interviews.update_one(
        {"id": interview_id},
        {"$set": update_data}
    )
    
    return ResponseWithEvaluation(
        id=response_id,
        question_id=question["id"],
        answer_text=answer_data.answer_text,
        evaluation=evaluation,
        timestamp=datetime.fromisoformat(response_doc["timestamp"]),
        next_question=next_question,
        interview_completed=interview_completed
    )

@api_router.get("/interviews", response_model=List[InterviewResponse])
async def get_interviews(current_user: dict = Depends(get_current_user)):
    interviews = []
    async for interview in db.interviews.find({"user_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1):
        interviews.append(InterviewResponse(
            id=interview["id"],
            user_id=interview["user_id"],
            status=interview["status"],
            settings=InterviewSettings(**interview["settings"]),
            current_question_index=interview["current_question_index"],
            total_score=interview["total_score"],
            created_at=datetime.fromisoformat(interview["created_at"]),
            completed_at=datetime.fromisoformat(interview["completed_at"]) if interview.get("completed_at") else None
        ))
    return interviews

@api_router.get("/interviews/{interview_id}/summary", response_model=InterviewSummary)
async def get_interview_summary(interview_id: str, current_user: dict = Depends(get_current_user)):
    interview = await db.interviews.find_one({"id": interview_id, "user_id": current_user["id"]})
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    if interview["status"] != "completed":
        raise HTTPException(status_code=400, detail="Interview not completed yet")
    
    # Get all responses
    responses = []
    async for resp in db.responses.find({"interview_id": interview_id}):
        responses.append(resp)
    
    # Calculate statistics
    total_questions = len(responses)
    total_score = interview["total_score"]
    max_score = total_questions * 20
    avg_score = total_score / total_questions if total_questions > 0 else 0
    
    # Collect strengths and improvements
    all_strengths = []
    all_improvements = []
    competency_scores = {}
    
    for resp in responses:
        eval_data = resp["evaluation"]
        all_strengths.extend(eval_data["strengths"])
        all_improvements.extend(eval_data["improvements"])
        
        # Get question to know competency
        question = await db.questions.find_one({"id": resp["question_id"]})
        comp = question["competency"]
        
        if comp not in competency_scores:
            competency_scores[comp] = {"total": 0, "count": 0, "scores": []}
        
        competency_scores[comp]["total"] += eval_data["scores"]["total"]
        competency_scores[comp]["count"] += 1
        competency_scores[comp]["scores"].append(eval_data["scores"]["total"])
    
    # Get top strengths and improvements
    top_strengths = list(set(all_strengths))[:5]
    top_improvements = list(set(all_improvements))[:5]
    
    # Calculate competency breakdown
    competency_breakdown = {}
    for comp, data in competency_scores.items():
        avg = data["total"] / data["count"] if data["count"] > 0 else 0
        competency_breakdown[comp] = {
            "average_score": round(avg, 1),
            "questions_asked": data["count"],
            "performance": "Strong" if avg >= 16 else "Good" if avg >= 12 else "Needs Improvement"
        }
    
    # Determine readiness level
    avg_percentage = (avg_score / 20) * 100
    if avg_percentage >= 80:
        readiness = "Excellent - Interview Ready"
    elif avg_percentage >= 65:
        readiness = "Good - Minor improvements needed"
    elif avg_percentage >= 50:
        readiness = "Fair - Practice recommended"
    else:
        readiness = "Needs Significant Improvement"
    
    # Personalized advice
    advice = [
        "Always structure your answers using the STAR framework (Situation, Task, Action, Result)",
        "Be specific with examples - avoid vague or general statements",
        "Quantify results whenever possible (e.g., 'increased efficiency by 30%')"
    ]
    
    if avg_score < 12:
        advice.append("Focus on providing more detailed context about the situations you describe")
    if any("action" in imp.lower() for imp in all_improvements):
        advice.append("Emphasize YOUR specific actions and contributions, not just team efforts")
    
    return InterviewSummary(
        interview_id=interview_id,
        total_questions=total_questions,
        overall_score=total_score,
        max_score=max_score,
        average_score=round(avg_score, 1),
        top_strengths=top_strengths,
        top_improvements=top_improvements,
        competency_breakdown=competency_breakdown,
        readiness_level=readiness,
        personalized_advice=advice
    )

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
