"""
Planner Agent - Architecture and task decomposition
Breaks down ideas into executable tasks
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from src.agents.base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from src.core.mode_manager import SystemMode


@dataclass
class TaskNode:
    """Node in task decomposition graph"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_time: float = 0.0  # in minutes
    priority: int = 1  # 1-5, higher is more urgent
    assigned_to: Optional[str] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectPlan:
    """Complete project plan"""
    name: str
    description: str
    tasks: List[TaskNode] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_total_time: float = 0.0
    architecture: Dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: List[str] = field(default_factory=list)


class PlannerAgent(BaseAgent):
    """
    Planner Agent - Decomposes ideas into executable tasks
    
    Responsibilities:
    1. Analyze requirements
    2. Create architecture
    3. Decompose into tasks
    4. Define dependencies
    5. Set acceptance criteria
    """
    
    def __init__(self):
        super().__init__(
            name="planner",
            capabilities=[
                AgentCapability.PLANNING,
                AgentCapability.CODE_READING
            ]
        )
        
        # Planning templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load planning templates"""
        return {
            "web_app": {
                "phases": ["setup", "backend", "frontend", "database", "deployment"],
                "default_tasks": [
                    "Set up project structure",
                    "Configure dependencies",
                    "Implement core functionality",
                    "Add testing",
                    "Document code"
                ]
            },
            "cli_tool": {
                "phases": ["design", "core", "parsing", "io", "packaging"],
                "default_tasks": [
                    "Define command interface",
                    "Implement core logic",
                    "Add argument parsing",
                    "Handle file I/O",
                    "Create packaging"
                ]
            },
            "library": {
                "phases": ["api", "implementation", "testing", "documentation", "distribution"],
                "default_tasks": [
                    "Design public API",
                    "Implement core classes/functions",
                    "Write unit tests",
                    "Create documentation",
                    "Setup distribution"
                ]
            }
        }
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute planning based on context
        
        Steps:
        1. Analyze requirements
        2. Determine project type
        3. Create architecture
        4. Decompose into tasks
        5. Define dependencies
        """
        self.logger.info(f"Starting planning for task: {context.task_id}")
        
        try:
            # Step 1: Analyze requirements
            requirements = await self._analyze_requirements(context)
            
            # Step 2: Determine project type
            project_type = await self._determine_project_type(requirements)
            
            # Step 3: Create architecture
            architecture = await self._create_architecture(requirements, project_type)
            
            # Step 4: Decompose into tasks
            tasks = await self._decompose_tasks(requirements, architecture, project_type)
            
            # Step 5: Define dependencies
            dependency_graph = await self._build_dependency_graph(tasks)
            
            # Step 6: Create complete plan
            plan = ProjectPlan(
                name=requirements.get("project_name", "Unnamed Project"),
                description=requirements.get("description", ""),
                tasks=tasks,
                dependencies=dependency_graph,
                estimated_total_time=sum(t.estimated_time for t in tasks),
                architecture=architecture,
                acceptance_criteria=requirements.get("acceptance_criteria", [])
            )
            
            # Step 7: Store plan in state
            await self._store_plan(context, plan)
            
            # Step 8: Create tasks in state manager
            await self._create_state_tasks(context, tasks)
            
            return AgentResponse(
                success=True,
                message=self._format_plan_message(plan),
                data={
                    "plan": self._plan_to_dict(plan),
                    "has_plan": True,
                    "plan_complete": True,
                    "task_count": len(tasks),
                    "estimated_time": plan.estimated_total_time
                },
                next_step="Switch to /dev mode to start implementation"
            )
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Planning failed: {str(e)}",
                error=str(e),
                requires_human_input=True
            )
    
    async def _analyze_requirements(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze requirements from user input"""
        prompt = f"""
        Analyze the following project requirements and extract key information:
        
        User Input: {context.user_input or context.task_description}
        
        Previous Context: {json.dumps(context.previous_results, indent=2) if context.previous_results else "None"}
        
        Extract:
        1. Project name or title
        2. Main goal/purpose
        3. Key features needed
        4. Technology preferences (if mentioned)
        5. Constraints (time, resources, etc.)
        6. Acceptance criteria
        7. Project type (web app, CLI, library, etc.)
        
        Return as JSON with those keys.
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a software requirements analyzer."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            requirements = json.loads(response.content)
            return requirements
        except:
            # Fallback parsing
            return {
                "project_name": "New Project",
                "description": context.task_description,
                "features": ["Basic functionality"],
                "acceptance_criteria": ["It works"],
                "project_type": "general"
            }
    
    async def _determine_project_type(self, requirements: Dict[str, Any]) -> str:
        """Determine project type from requirements"""
        description = requirements.get("description", "").lower()
        features = requirements.get("features", [])
        
        # Check for keywords
        if any(word in description for word in ["web", "website", "app", "api", "rest"]):
            return "web_app"
        elif any(word in description for word in ["cli", "command", "terminal", "shell"]):
            return "cli_tool"
        elif any(word in description for word in ["library", "package", "module", "sdk"]):
            return "library"
        elif any(word in description for word in ["data", "analysis", "processing"]):
            return "data_processing"
        else:
            return "general"
    
    async def _create_architecture(
        self, 
        requirements: Dict[str, Any], 
        project_type: str
    ) -> Dict[str, Any]:
        """Create high-level architecture"""
        prompt = f"""
        Create a software architecture for the following project:
        
        Project: {requirements.get('project_name')}
        Description: {requirements.get('description')}
        Type: {project_type}
        Features: {requirements.get('features', [])}
        
        Provide a JSON response with:
        1. components: List of main components/modules
        2. technologies: Suggested tech stack
        3. patterns: Architectural patterns to use
        4. file_structure: Recommended directory structure
        5. interfaces: Key interfaces/APIs
        
        Be specific and practical.
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a software architect."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            architecture = json.loads(response.content)
            
            # Add default structure based on project type
            template = self.templates.get(project_type, {})
            if template:
                architecture["phases"] = template.get("phases", [])
            
            return architecture
        except:
            # Return basic architecture
            return {
                "components": ["Main module"],
                "technologies": ["Python"],
                "patterns": ["Modular"],
                "file_structure": ["src/", "tests/", "docs/"],
                "phases": ["setup", "development", "testing"]
            }
    
    async def _decompose_tasks(
        self, 
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        project_type: str
    ) -> List[TaskNode]:
        """Decompose project into individual tasks"""
        prompt = f"""
        Break down this project into concrete, executable tasks:
        
        Project: {requirements.get('project_name')}
        Architecture: {json.dumps(architecture, indent=2)}
        Features: {json.dumps(requirements.get('features', []), indent=2)}
        
        Create tasks with:
        1. Clear, actionable description
        2. Estimated time (in minutes)
        3. Priority (1-5)
        4. Dependencies (list of task IDs it depends on)
        
        Return as a JSON list of task objects, each with:
        - id: Unique identifier (e.g., "task_001")
        - description: What to do
        - estimated_time: Minutes to complete
        - priority: 1-5
        - dependencies: List of task IDs
        - metadata: Any additional info
        
        Make tasks small enough to complete in one sitting (15-120 minutes).
        Include setup, implementation, testing, and documentation tasks.
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a task decomposition expert."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            tasks_data = json.loads(response.content)
            tasks = []
            
            for i, task_data in enumerate(tasks_data, 1):
                task = TaskNode(
                    id=f"task_{i:03d}",
                    description=task_data.get("description", f"Task {i}"),
                    estimated_time=task_data.get("estimated_time", 30.0),
                    priority=task_data.get("priority", 3),
                    dependencies=task_data.get("dependencies", []),
                    metadata=task_data.get("metadata", {})
                )
                tasks.append(task)
            
            return tasks
        except:
            # Create basic tasks from template
            template = self.templates.get(project_type, self.templates["web_app"])
            tasks = []
            
            for i, task_desc in enumerate(template["default_tasks"], 1):
                tasks.append(TaskNode(
                    id=f"task_{i:03d}",
                    description=task_desc,
                    estimated_time=30.0,
                    priority=3,
                    dependencies=[] if i == 1 else [f"task_{i-1:03d}"]
                ))
            
            return tasks
    
    async def _build_dependency_graph(self, tasks: List[TaskNode]) -> List[str]:
        """Build dependency graph from tasks"""
        # Simple linear dependency for now
        # Can be enhanced with topological sorting
        dependencies = []
        
        for i, task in enumerate(tasks):
            for dep_id in task.dependencies:
                dependencies.append(f"{dep_id} -> {task.id}")
        
        return dependencies
    
    async def _store_plan(self, context: AgentContext, plan: ProjectPlan):
        """Store plan in system state"""
        # Convert plan to dict
        plan_dict = self._plan_to_dict(plan)
        
        # Store in state manager
        self.state_manager.log_action(
            agent=self.name,
            action="plan_created",
            details={
                "task_id": context.task_id,
                "plan": plan_dict,
                "task_count": len(plan.tasks)
            }
        )
        
        # Also store in file context
        plan_json = json.dumps(plan_dict, indent=2)
        self.state_manager.add_file_context(
            f".plan_{context.task_id}.json",
            plan_json
        )
    
    async def _create_state_tasks(self, context: AgentContext, tasks: List[TaskNode]):
        """Create tasks in state manager"""
        for task in tasks:
            self.state_manager.create_task(
                description=task.description,
                metadata={
                    "planner_task_id": task.id,
                    "estimated_time": task.estimated_time,
                    "priority": task.priority,
                    "dependencies": task.dependencies,
                    **task.metadata
                }
            )
    
    def _plan_to_dict(self, plan: ProjectPlan) -> Dict[str, Any]:
        """Convert plan to dictionary"""
        return {
            "name": plan.name,
            "description": plan.description,
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "estimated_time": task.estimated_time,
                    "priority": task.priority,
                    "dependencies": task.dependencies,
                    "metadata": task.metadata
                }
                for task in plan.tasks
            ],
            "dependencies": plan.dependencies,
            "estimated_total_time": plan.estimated_total_time,
            "architecture": plan.architecture,
            "acceptance_criteria": plan.acceptance_criteria
        }
    
    def _format_plan_message(self, plan: ProjectPlan) -> str:
        """Format plan as readable message"""
        lines = [
            f"**Project Plan: {plan.name}**",
            f"Description: {plan.description}",
            "",
            f"**Tasks ({len(plan.tasks)} total, {plan.estimated_total_time:.0f} minutes estimated):**"
        ]
        
        for task in plan.tasks:
            lines.append(
                f"  - {task.id}: {task.description} "
                f"({task.estimated_time:.0f}m, priority {task.priority})"
            )
            if task.dependencies:
                lines.append(f"    Depends on: {', '.join(task.dependencies)}")
        
        lines.extend([
            "",
            "**Next Steps:**",
            "1. Review the plan above",
            "2. Use /dev mode to start implementation",
            "3. Tasks will be executed in dependency order"
        ])
        
        return "\n".join(lines)
