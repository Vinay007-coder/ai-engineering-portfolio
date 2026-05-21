# Complete Pydantic v2 Practice Script

# ------------------------------------------------------------
# Pydantic v2 Features Demonstrated
# ------------------------------------------------------------

# 1. Pydantic v2 Schemas
#
#    - Defines structured data models
#      using Python classes.
#
#    - Enforces:
#         • type validation
#         • schema consistency
#         • automatic parsing
#
#    - Helps create reliable API contracts.

# 2. Optional Fields
#
#    - Allows certain fields
#      to be omitted or set to None.
#
#    - Uses:
#
#          Optional[type]
#
#    - Example:
#
#          Optional[str]
#
#    - Useful for:
#         • partial updates
#         • flexible request payloads
#         • non-required metadata

# 3. Whitespace Stripping
#    via @field_validator
#
#    - Automatically cleans input values
#      before storing them.
#
#    - Removes:
#         • leading spaces
#         • trailing spaces
#
#    - Ensures normalized and clean data.
#
#    - Example use case:
#
#          "   hello   "
#
#      becomes:
#
#          "hello"

# 4. Parsing with model_validate()
#
#    - Converts raw external data
#      into validated Pydantic objects.
#
#    - Performs:
#         • type coercion
#         • schema validation
#         • data normalization
#
#    - Commonly used when handling:
#         • API requests
#         • JSON payloads
#         • database records

# 5. Serialization with model_dump()
#
#    - Converts validated models
#      back into standard Python dictionaries.
#
#    - Produces clean structured output
#      suitable for:
#         • API responses
#         • JSON conversion
#         • logging
#         • storage
#
#    - Helps maintain predictable output formatting.


from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

# ==========================================
# 1. NESTED MODELS & REUSABLE SUB-STRUCTURES
# ==========================================
class SentimentMetadata(BaseModel):
    # Optional field with a default value of None if missing
    model_version: Optional[str] = Field(default=None, description="The specific ML model identifier.")
    processing_time_ms: float = Field(gt=0, description="Inbound latency trace measurement.")

class SentimentScore(BaseModel):
    label: str = Field(description="Classification label like 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability metric score.")

# ==========================================
# 2. REQUEST MODEL WITH CUSTOM VALIDATION
# ==========================================
class SentimentRequest(BaseModel):
    # Field setup with default value, validation, and metadata description
    prompt: str = Field(min_length=3, max_length=500, description="The raw target sentence to analyze.")
    user_id: Optional[str] = Field(default=None, description="Optional tracking ID for billing profiles.")

    # Custom validator to clean text inputs automatically
    @field_validator('prompt')
    @classmethod
    def clean_and_strip_prompt(cls, value: str) -> str:
        # Enforce white space removal transformation
        cleaned_value = value.strip()
        if len(cleaned_value) < 3:
            raise ValueError("Prompt must contain at least 3 characters after stripping whitespace.")
        return cleaned_value

# ==========================================
# 3. RESPONSE MODEL WITH NESTED RELATIONSHIPS
# ==========================================
class SentimentResponse(BaseModel):
    original_prompt: str
    top_prediction: SentimentScore  # Single Nested Pydantic Model
    all_scores: List[SentimentScore] = Field(default=[])  # List of Nested Models
    metadata: SentimentMetadata  # Nested Lifecycle tracking structure


# ==========================================
# 4. TESTING VALIDATION AND SERIALIZATION FLOWS
# ==========================================
if __name__ == "__main__":
    print("--- 1. Testing Input Parsing and @field_validator ---")
    
    # Raw input data simulation (notice the messy padding and missing optional user_id)
    raw_api_request = {
        "prompt": "   The product quality is absolutely stunning!   "
    }

    # model.model_validate(): Parse from python dictionary
    validated_request = SentimentRequest.model_validate(raw_api_request)
    
    print(f"Original Text JSON Length: {len(raw_api_request['prompt'])}")
    print(f"Validated Text Model Length: {len(validated_request.prompt)}")
    print(f"Cleaned prompt value: '{validated_request.prompt}'")
    print(f"Optional user_id value default status: {validated_request.user_id}")


    print("\n--- 2. Building and Validating Complex Response Models ---")
    
    # Mocking data to represent an output payload from an ML server engine
    raw_engine_output = {
        "original_prompt": validated_request.prompt,
        "top_prediction": {"label": "POSITIVE", "confidence": 0.98},
        "all_scores": [
            {"label": "POSITIVE", "confidence": 0.98},
            {"label": "NEGATIVE", "confidence": 0.01},
            {"label": "NEUTRAL", "confidence": 0.01}
        ],
        "metadata": {
            "model_version": "distilbert-v4-sentiment",
            "processing_time_ms": 12.45
        }
    }

    # Parse dictionary into complex response nested system
    validated_response = SentimentResponse.model_validate(raw_engine_output)
    print(f"Successfully processed model labels. Top category: {validated_response.top_prediction.label}")
    print(f"Metadata model tracker reference: {validated_response.metadata.model_version}")


    print("\n--- 3. Testing model.model_dump() Serialization ---")
    
    # model.model_dump(): Serialize your active memory model instances back to clear native dictionaries
    clean_dict_output = validated_response.model_dump()
    
    print(f"Serialized output type check: {type(clean_dict_output)}")
    print("Clean dictionary serialization preview layout:")
    print(clean_dict_output)

# --------------------------------------------
# Step-by-Step Execution Flow
# --------------------------------------------

# ------------------------------------------------------------
# 1. Inbound Parsing and Transformation
#    (model_validate)
# ------------------------------------------------------------

# When:
#
#     SentimentRequest.model_validate(raw_api_request)
#
# is executed, Pydantic immediately begins
# validating and transforming incoming data.

# The "prompt" field is routed through:
#
#     @field_validator('prompt')
#
# before being stored inside the model.

# Inside the validator:
#
#     .strip()
#
# removes leading and trailing whitespace.

# Example:
#
#     "    Explain AI systems    "
#
# becomes:
#
#     "Explain AI systems"

# The prompt length therefore shrinks
# from 48 characters down to 41 characters
# before persistence.

# ------------------------------------------------------------
# Optional Field Resolution
# ------------------------------------------------------------

# The raw request payload does not contain:
#
#     user_id
#
# Pydantic checks the schema definition:
#
#     Optional[str] = Field(default=None)
#
# Since the field is optional,
# validation succeeds safely.
#
# Pydantic automatically assigns:
#
#     user_id = None
#
# instead of raising a validation error.

# ------------------------------------------------------------
# 2. Resolving Hierarchies
#    (Nested Validation)
# ------------------------------------------------------------

# While validating:
#
#     SentimentResponse
#
# Pydantic processes fields sequentially.

# It reaches:
#
#     top_prediction
#
# and detects that it must match:
#
#     SentimentScore
#
# Pydantic immediately validates
# the nested payload:
#
#     {
#         "label": "POSITIVE",
#         "confidence": 0.98
#     }
#
# against the child model schema.

# This ensures:
#
#     • correct field names
#     • proper data types
#     • valid nested structure
#
# before allowing model creation.

# ------------------------------------------------------------
# 3. Output Serialization
#    (model_dump)
# ------------------------------------------------------------

# The:
#
#     .model_dump()
#
# method converts typed Pydantic objects
# into standard Python dictionaries.

# Complex model structures are flattened into:
#
#     • primitive types
#     • lists
#     • dictionaries
#
# making the data safe for:
#
#     • database storage
#     • JSON serialization
#     • API responses
#     • logging systems

# The resulting output contains only
# raw serializable Python data structures.