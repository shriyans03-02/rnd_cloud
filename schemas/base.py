from pydantic import BaseModel, model_validator


class TrimmedModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def trim_strings(cls, data):
        if isinstance(data, dict):
            return {
                k: v.strip() if isinstance(v, str) else v
                for k, v in data.items()
            }
        return data
