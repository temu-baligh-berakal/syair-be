from fastapi import APIRouter, HTTPException, status
from app.schemas.hadits_schema import LLMSummarizerRequest
from app.services.llm_summarizer_service import summarize_hadits

router = APIRouter()

@router.post("/summarize", response_model=str, status_code=status.HTTP_200_OK)
async def get_summary(request: LLMSummarizerRequest):
    """Meringkas beberapa hadits terkait dengan suatu query menggunakan LLM."""
    try:
        summary = summarize_hadits(request)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}",
        )
