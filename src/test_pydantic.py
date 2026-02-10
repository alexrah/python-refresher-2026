from pydantic import field_validator, BaseModel
from typing import Optional, List

class User(BaseModel):
    name: str
    email: str

    @field_validator('email')
    def validate_email(cls, v:str):
        if '@' not in v:
            raise ValueError('invalid email')
        return v.lower()


def main():
    user = User(
        name = 'Foo',
        email = "fixPAX.COM"
    )
    print(user)



if __name__ == "__main__":
    main()


