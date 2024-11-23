"""Define the configurable parameters and schemas for the agent."""
from dataclasses import dataclass, field, fields
from decimal import Decimal
from typing import Annotated, Dict, Optional, Any, List
from pydantic import BaseModel, Field
from django.conf import settings
from langchain_core.runnables import RunnableConfig, ensure_config


class BaseOrderSchema(BaseModel):
    """Base schema for order-related tools"""
    order_id: str = Field(description="The ID of the order")
    customer_id: str = Field(description="The ID of the customer")


class TrackingRequest(BaseOrderSchema):
    """Schema for tracking-related tools"""
    include_history: bool = Field(
        default=False,
        description="Whether to include full tracking history"
    )


class ModifyOrderQuantity(BaseOrderSchema):
    """Schema for modifying order quantity"""
    product_id: int = Field(description="The ID of the product to modify")
    new_quantity: int = Field(description="The new quantity desired")


class CancelOrderRequest(BaseOrderSchema):
    """Schema for order cancellation"""
    reason: str = Field(description="The reason for cancellation")


@dataclass
class Product:
    id: str
    name: str
    price: Decimal
    category: str
    description: str
    stock: int


@dataclass
class CartItem:
    product_id: str
    quantity: int
    price: Decimal


@dataclass
class UserCart:
    user_id: str
    items: List[CartItem]
    total: Decimal


# Agent Configuration
@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default="You are a helpful customer support agent specializing in order management...",
        metadata={
            "description": "The system prompt to use for the agent's interactions."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default=f"openai/'gpt-4o-mini",
        default=f"openai/{settings.GPT_MINI}",
        metadata={
            "description": "The language model to use for the agent's interactions."
        },
    )

    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return."
        },
    )

    tool_timeout: int = field(
        default=30,
        metadata={
            "description": "Maximum time in seconds for tool execution."
        },
    )

    max_tool_retries: int = field(
        default=3,
        metadata={
            "description": "Maximum number of retries for failed tool executions."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_search_results": self.max_search_results,
            "tool_timeout": self.tool_timeout,
            "max_tool_retries": self.max_tool_retries,
        }


# Tool-specific configurations
@dataclass
class ToolConfig:
    """Configuration for individual tools."""
    requires_confirmation: bool = False
    max_retries: int = field(
        default_factory=lambda: Configuration().max_tool_retries)
    timeout: int = field(default_factory=lambda: Configuration().tool_timeout)


if __name__ == '__main__':
    # Example usage
    config = Configuration()
    print(f"Default Configuration: {config.to_dict()}")

    # Example of creating tool config
    tracking_config = ToolConfig()
    print(f"Default Tool Config: {tracking_config}")
