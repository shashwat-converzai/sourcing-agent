"""
Anthropic Claude service for summarizing long candidate resumes.
Uses async parallel API calls for efficient summarization.
"""
from anthropic import Anthropic
from typing import List
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from ..config import Settings

logger = logging.getLogger(__name__)

# Create a summary logger that only writes to file (no console)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"ingestion_{datetime.now().strftime('%Y%m%d')}.log"

summary_logger = logging.getLogger("summaries")
summary_logger.setLevel(logging.INFO)
# Remove any existing handlers to avoid duplicates
if not summary_logger.handlers:
    summary_file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    summary_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    summary_logger.addHandler(summary_file_handler)
    summary_logger.propagate = False  # Don't propagate to root logger to avoid console output


class ResumeSummarizerService:
    """Service for summarizing long resumes using Anthropic Claude with async parallel processing."""
    
    SYSTEM_PROMPT = """You are an expert resume information extraction and restructuring model.

Your job is to transform long resumes into a structured format with light compression.  

Do NOT deeply summarize. Do NOT add or invent information.  

Reorganize the content and remove redundancies while keeping all important factual details.

OUTPUT SECTIONS (use exactly these, in this order):

1. [PROFESSIONAL_SUMMARY]

2. [OBJECTIVE]

3. [CORE_SKILLS_AND_TECHNOLOGIES]

4. [RECENT_EXPERIENCE]

5. [EARLIER_EXPERIENCE]

6. [MATCHING_SIGNALS_FOR_JOB_DESCRIPTION]

SECTION REQUIREMENTS:

[PROFESSIONAL_SUMMARY]

- 3–5 sentences describing experience level, domains, functional areas, and specialization.

- Use only information present in the resume.

[OBJECTIVE]

- Extract only the explicit objective statement. Do not rewrite or embellish.

[CORE_SKILLS_AND_TECHNOLOGIES]

- Combine all skills, tools, and technologies into one deduplicated list.

- Keep exact technical terms.

- Remove filler and soft skills.

[RECENT_EXPERIENCE]

- Include the 3–5 most recent roles.

- For each: Title, Employer (PII removed), Years.

- Add 2–4 condensed bullets summarizing responsibilities, outcomes, and key tools.

- Merge repeated statements.

[EARLIER_EXPERIENCE]

- Provide a short consolidated summary of older roles, responsibilities, domains, and seniority.

[MATCHING_SIGNALS_FOR_JOB_DESCRIPTION]

- Provide 4–8 bullet points highlighting match-relevant attributes.

- These MUST be grounded strictly in the resume text.

- Focus on domain fit, module expertise, tools, implementation vs support, relevant responsibilities, and seniority markers.

- No opinions, no soft skills, no invented strengths.

GLOBAL RULES:

- Remove ALL PII: names, emails, phone numbers, LinkedIn URLs, physical addresses.

- Keep job titles, tools, responsibilities, and technologies EXACT.

- Do not add, guess, or hallucinate.

- Deduplicate repeated content.

- Light compression only; final text must remain under 1,500 tokens.

- Maintain chronological clarity.

- Output only the sections listed above."""
    
    USER_PROMPT_TEMPLATE = """Restructure and compress into required sections. Preserve facts, remove PII, deduplicate. Keep technical details. Output under 1,500 tokens. No additions.

[PROFESSIONAL_SUMMARY]
[OBJECTIVE]
[CORE_SKILLS_AND_TECHNOLOGIES]
[RECENT_EXPERIENCE]
[EARLIER_EXPERIENCE]
[MATCHING_SIGNALS_FOR_JOB_DESCRIPTION]

--------------------
{resume_text}
--------------------"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required but not set")
        
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        self.batch_size = settings.anthropic_batch_size
        self.max_workers = settings.anthropic_max_workers
        
        logger.info(f"Initialized Resume Summarizer with model: {self.model}, batch_size: {self.batch_size}, max_workers: {self.max_workers} (async parallel processing)")
    
    def _summarize_sync(self, resume_text: str) -> str:
        """
        Synchronous wrapper for summarizing a long resume text using Anthropic Claude.
        This is called from async context using run_in_executor.
        
        Args:
            resume_text: The full resume text to summarize
            
        Returns:
            Summarized resume text in structured format (compressed to under 1500 tokens)
        """
        if not resume_text or not resume_text.strip():
            return resume_text
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(resume_text=resume_text)
        
        max_retries = 3  # Total of 4 attempts: initial + 3 retries
        wait_times = [5, 10, 30]  # Wait times in seconds for each retry
        
        for attempt in range(max_retries + 1):
            try:
                # Use prompt caching for the system prompt (static content)
                # The system prompt is the same for all requests, so we cache it with ephemeral cache
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,  # Limit output to 1500 tokens
                    temperature=0.1,  # Low temperature for factual extraction
                    system=[
                        {
                            "type": "text",
                            "text": self.SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"}  # Cache the system prompt
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )
                
                summary = message.content[0].text.strip()
                summary_logger.debug(f"Summarized resume: {len(resume_text)} chars -> {len(summary)} chars")
                summary_logger.info(f"Summarized resume: {summary}")
                return summary
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str or 
                    "429" in error_str or
                    "too many requests" in error_str or
                    "quota" in error_str
                )
                
                if attempt < max_retries and (is_rate_limit or True):  # Retry on any error
                    wait_time = wait_times[attempt]  # Get wait time for this attempt
                    summary_logger.warning(f"Error summarizing resume (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    summary_logger.error(f"Error summarizing resume after {attempt + 1} attempts: {e}")
                    # Return None to indicate failure
                    return None
    
    async def summarize_batch(self, resume_texts: List[str]) -> List[str]:
        """
        Summarize multiple resume texts using async parallel API calls.
        Processes exactly 128 texts at a time with parallel API calls using ThreadPoolExecutor.
        
        Args:
            resume_texts: List of resume texts to summarize (should be exactly 128 or less)
            
        Returns:
            List of summarized texts (same order as input), with None for failed summaries
        """
        if not resume_texts:
            return []
        
        # Limit to 128 texts
        texts_chunk = resume_texts[:128]
        
        summary_logger.info(f"Processing summarization batch: {len(texts_chunk)} texts")
        
        try:
            # Use ThreadPoolExecutor for parallel API calls
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(texts_chunk))) as executor:
                # Create tasks for all texts in this chunk
                tasks = [
                    loop.run_in_executor(executor, self._summarize_sync, text)
                    for text in texts_chunk
                ]
                
                # Wait for all tasks to complete
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and validate
                validated_results = []
                for idx, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        summary_logger.error(f"Error summarizing resume at index {idx}: {result}")
                        validated_results.append(None)
                    elif result is None:
                        summary_logger.warning(f"Summarization returned None for index {idx}")
                        validated_results.append(None)
                    elif not result.strip():
                        summary_logger.warning(f"Summarization returned empty result for index {idx}")
                        validated_results.append(None)
                    elif result.strip() == texts_chunk[idx].strip():
                        summary_logger.warning(f"Summarization returned same as original for index {idx}")
                        validated_results.append(None)
                    else:
                        validated_results.append(result)
                
                successful = len([r for r in validated_results if r is not None])
                summary_logger.info(f"Completed summarization batch: {successful}/{len(texts_chunk)} summaries generated")
                return validated_results
                    
        except Exception as e:
            summary_logger.error(f"Error processing summarization batch: {e}")
            # Mark all in this batch as failed
            return [None] * len(texts_chunk)

