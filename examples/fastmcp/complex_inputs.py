"""
FastMCP Complex inputs Example

Demonstrates validation via pydantic with complex models,
and structured output for returning validated data.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Shrimp Tank")


class ShrimpTank(BaseModel):
    class Shrimp(BaseModel):
        name: Annotated[str, Field(max_length=10)]
        color: str = "red"
        age_days: int = 0

    shrimp: list[Shrimp]
    temperature: float = Field(default=24.0, ge=20.0, le=28.0)
    ph_level: float = Field(default=7.0, ge=6.5, le=8.0)


@mcp.tool()
def name_shrimp(
    tank: ShrimpTank,
    # You can use pydantic Field in function signatures for validation.
    extra_names: Annotated[list[str], Field(max_length=10)],
) -> list[str]:
    """List all shrimp names in the tank"""
    return [shrimp.name for shrimp in tank.shrimp] + extra_names


# Structured output example - returns a validated tank analysis
class TankAnalysis(BaseModel):
    """Analysis of shrimp tank conditions"""
    total_shrimp: int
    temperature_status: str  # "optimal", "too_cold", "too_hot"
    ph_status: str  # "optimal", "too_acidic", "too_basic"
    shrimp_by_color: dict[str, int]
    oldest_shrimp: str | None
    recommendations: list[str]


@mcp.tool(structured_output=True)
def analyze_tank(tank: ShrimpTank) -> TankAnalysis:
    """Analyze tank conditions and provide recommendations"""
    # Temperature analysis
    if tank.temperature < 22:
        temp_status = "too_cold"
    elif tank.temperature > 26:
        temp_status = "too_hot"
    else:
        temp_status = "optimal"
    
    # pH analysis
    if tank.ph_level < 6.8:
        ph_status = "too_acidic"
    elif tank.ph_level > 7.5:
        ph_status = "too_basic"
    else:
        ph_status = "optimal"
    
    # Count shrimp by color
    color_counts: dict[str, int] = {}
    for shrimp in tank.shrimp:
        color_counts[shrimp.color] = color_counts.get(shrimp.color, 0) + 1
    
    # Find oldest shrimp
    oldest = None
    if tank.shrimp:
        oldest_shrimp_obj = max(tank.shrimp, key=lambda s: s.age_days)
        oldest = oldest_shrimp_obj.name
    
    # Generate recommendations
    recommendations = []
    if temp_status == "too_cold":
        recommendations.append("Increase water temperature to 22-26°C")
    elif temp_status == "too_hot":
        recommendations.append("Decrease water temperature to 22-26°C")
    
    if ph_status == "too_acidic":
        recommendations.append("Add crushed coral or baking soda to raise pH")
    elif ph_status == "too_basic":
        recommendations.append("Add Indian almond leaves or driftwood to lower pH")
    
    if len(tank.shrimp) > 20:
        recommendations.append("Consider dividing colony to prevent overcrowding")
    
    if not recommendations:
        recommendations.append("Tank conditions are optimal!")
    
    return TankAnalysis(
        total_shrimp=len(tank.shrimp),
        temperature_status=temp_status,
        ph_status=ph_status,
        shrimp_by_color=color_counts,
        oldest_shrimp=oldest,
        recommendations=recommendations
    )


# Another structured output example - breeding recommendations
@mcp.tool(structured_output=True)
def get_breeding_pairs(tank: ShrimpTank) -> dict[str, list[str]]:
    """Suggest breeding pairs by color
    
    Returns a dictionary mapping colors to lists of shrimp names
    that could be bred together.
    """
    pairs_by_color: dict[str, list[str]] = {}
    
    for shrimp in tank.shrimp:
        if shrimp.age_days >= 60:  # Mature enough to breed
            if shrimp.color not in pairs_by_color:
                pairs_by_color[shrimp.color] = []
            pairs_by_color[shrimp.color].append(shrimp.name)
    
    # Only return colors with at least 2 shrimp
    return {
        color: names 
        for color, names in pairs_by_color.items() 
        if len(names) >= 2
    }


if __name__ == "__main__":
    # For testing the tools
    import asyncio
    
    async def test():
        # Create a test tank
        tank = ShrimpTank(
            shrimp=[
                ShrimpTank.Shrimp(name="Rex", color="red", age_days=90),
                ShrimpTank.Shrimp(name="Blue", color="blue", age_days=45),
                ShrimpTank.Shrimp(name="Crimson", color="red", age_days=120),
                ShrimpTank.Shrimp(name="Azure", color="blue", age_days=80),
                ShrimpTank.Shrimp(name="Jade", color="green", age_days=30),
                ShrimpTank.Shrimp(name="Ruby", color="red", age_days=75),
            ],
            temperature=23.5,
            ph_level=7.2
        )
        
        # Test name_shrimp (non-structured output)
        names = name_shrimp(tank, ["Bonus1", "Bonus2"])
        print("Shrimp names:", names)
        
        # Test analyze_tank (structured output)
        print("\nTank Analysis:")
        analysis = analyze_tank(tank)
        print(analysis.model_dump_json(indent=2))
        
        # Test get_breeding_pairs (structured output returning dict)
        print("\nBreeding Pairs:")
        pairs = get_breeding_pairs(tank)
        print(f"Mature shrimp by color: {pairs}")
        
        # Show the tools that would be exposed
        print("\nAvailable tools:")
        tools = await mcp.list_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            if tool.outputSchema:
                print(f"  Output schema: {tool.outputSchema.get('title', tool.outputSchema.get('type', 'structured'))}")
    
    asyncio.run(test())
