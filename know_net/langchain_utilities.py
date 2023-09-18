import enum
from typing import Dict, cast

from langchain.prompts import chat as chat_prompt


class Roles(str, enum.Enum):
    SYSTEM = "system"
    BOT = "assistant"
    USER = "user"


PROMPT_TEMPLATE_MAPPING = cast(
    Dict[Roles, chat_prompt.BaseStringMessagePromptTemplate],
    {
        Roles.SYSTEM: chat_prompt.SystemMessagePromptTemplate,
        Roles.USER: chat_prompt.HumanMessagePromptTemplate,
        Roles.BOT: chat_prompt.AIMessagePromptTemplate,
    },
)


def construct_chat_prompt(
    text: str, role: Roles, template_format: str = "jinja2"
) -> chat_prompt.BaseStringMessagePromptTemplate:
    prompt_template_cls = PROMPT_TEMPLATE_MAPPING[role]
    return prompt_template_cls.from_template(
        template=text.strip("\n"),
        template_format=template_format,
    )
