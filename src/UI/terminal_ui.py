"""
Rich Terminal UI for ASEA-X
Provides beautiful, interactive terminal interface
"""

import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty
import time

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

from src.core.state_manager import SystemState, TaskStatus
from src.core.mode_manager import SystemMode


@dataclass
class UIEvent:
    """UI Event for communication"""
    type: str  # "update", "message", "error", "progress", "command"
    data: Any
    timestamp: float = field(default_factory=time.time)


class TerminalUI:
    """
    Rich terminal-based UI for ASEA-X
    
    Features:
    1. Multi-panel layout
    2. Real-time updates
    3. Syntax highlighting
    4. Progress tracking
    5. Interactive commands
    6. Colorful output
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.console = Console()
        self.layout = Layout()
        self.live: Optional[Live] = None
        
        # Event queue for async updates
        self.event_queue = Queue()
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        
        # UI state
        self.messages: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.system_status: Dict[str, Any] = {}
        self.progress_data: Dict[str, float] = {}
        self.last_update = time.time()
        
        # Setup layout
        self._setup_layout()
        
        # Color themes
        self.themes = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "system": "blue",
            "user": "magenta",
            "agent": "green",
            "mode": "yellow"
        }
    
    def _setup_layout(self):
        """Setup the terminal layout"""
        # Create layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main area
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left area
        self.layout["left"].split_column(
            Layout(name="messages", ratio=3),
            Layout(name="progress", size=5)
        )
        
        # Split right area
        self.layout["right"].split_column(
            Layout(name="status", size=10),
            Layout(name="tasks", ratio=2)
        )
    
    def start(self):
        """Start the UI"""
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start live display
        self.live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            screen=False
        )
        
        with self.live:
            self._main_loop()
    
    def stop(self):
        """Stop the UI"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
        
        if self.live:
            self.live.stop()
    
    def post_event(self, event: UIEvent):
        """Post an event to the UI"""
        self.event_queue.put(event)
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                # Process events
                while True:
                    try:
                        event = self.event_queue.get_nowait()
                        self._handle_event(event)
                    except Empty:
                        break
                
                # Update UI components
                self._update_header()
                self._update_messages()
                self._update_status()
                self._update_tasks()
                self._update_progress()
                self._update_footer()
                
                time.sleep(0.25)  # Update 4 times per second
                
            except Exception as e:
                self.console.print(f"[red]UI Update Error: {e}[/red]")
                time.sleep(1)
    
    def _handle_event(self, event: UIEvent):
        """Handle UI events"""
        if event.type == "message":
            self.messages.append({
                "type": "message",
                "content": event.data.get("content", ""),
                "sender": event.data.get("sender", "system"),
                "timestamp": event.timestamp
            })
            
            # Keep only last 50 messages
            if len(self.messages) > 50:
                self.messages = self.messages[-50:]
        
        elif event.type == "error":
            self.messages.append({
                "type": "error",
                "content": event.data.get("content", ""),
                "sender": event.data.get("sender", "system"),
                "timestamp": event.timestamp
            })
        
        elif event.type == "update":
            if "status" in event.data:
                self.system_status.update(event.data["status"])
            if "tasks" in event.data:
                self.tasks = event.data["tasks"]
            if "progress" in event.data:
                self.progress_data.update(event.data["progress"])
        
        elif event.type == "command":
            # Handle command input
            pass
    
    def _main_loop(self):
        """Main UI loop for input handling"""
        while self.running:
            try:
                # Get user input
                prompt = self._get_input_prompt()
                user_input = Prompt.ask(prompt)
                
                if user_input.strip().lower() in ['exit', 'quit', 'q']:
                    self.running = False
                    break
                
                # Post command event
                self.post_event(UIEvent(
                    type="command",
                    data={"input": user_input}
                ))
                
                # Echo user input
                self.post_event(UIEvent(
                    type="message",
                    data={
                        "content": user_input,
                        "sender": "user"
                    }
                ))
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Press Ctrl+C again to exit.[/yellow]")
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    self.running = False
                    break
            except EOFError:
                self.running = False
                break
            except Exception as e:
                self.console.print(f"[red]Input error: {e}[/red]")
    
    def _get_input_prompt(self) -> str:
        """Get formatted input prompt"""
        if not self.orchestrator:
            return "[cyan]asea-x>[/cyan] "
        
        state = self.orchestrator.state_manager.get_state()
        mode = state.current_mode
        
        # Color code based on mode
        mode_colors = {
            "chat": "cyan",
            "planner": "green",
            "dev": "yellow",
            "debug": "red",
            "lint": "magenta"
        }
        
        color = mode_colors.get(mode, "white")
        return f"[{color}]{mode}[/{color}]> "
    
    def _update_header(self):
        """Update header panel"""
        title = Text("ASEA-X - Autonomous Software Engineering Agent", style="bold blue")
        subtitle = Text("Multi-Agent System for Software Development", style="dim")
        
        # Mode indicator
        if self.orchestrator:
            state = self.orchestrator.state_manager.get_state()
            mode_text = Text(f"Mode: {state.current_mode.upper()}", style="bold yellow")
            
            # Safety indicator
            safety_text = Text("🔒 SAFE", style="bold green") if state.safety_enabled else Text("⚠️ UNSAFE", style="bold red")
            
            header_content = Group(
                title,
                subtitle,
                Text(""),
                Columns([mode_text, safety_text])
            )
        else:
            header_content = Group(title, subtitle)
        
        self.layout["header"].update(
            Panel(
                header_content,
                border_style="blue",
                padding=(1, 2)
            )
        )
    
    def _update_messages(self):
        """Update messages panel"""
        messages_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            expand=True
        )
        
        messages_table.add_column("Time", width=8)
        messages_table.add_column("From", width=10)
        messages_table.add_column("Message", ratio=1)
        
        for msg in self.messages[-15:]:  # Show last 15 messages
            # Format time
            dt = datetime.fromtimestamp(msg["timestamp"])
            time_str = dt.strftime("%H:%M:%S")
            
            # Format sender with color
            sender = msg["sender"]
            sender_color = self.themes.get(sender, "white")
            sender_text = Text(sender, style=sender_color)
            
            # Format message with type-based styling
            content = msg["content"]
            if msg["type"] == "error":
                message_text = Text(content, style="red")
            elif sender == "user":
                message_text = Text(content, style="magenta")
            elif sender == "system":
                message_text = Text(content, style="blue")
            else:
                message_text = Text(content)
            
            # Truncate long messages
            if len(content) > 100:
                message_text = Text(content[:97] + "...")
            
            messages_table.add_row(time_str, sender_text, message_text)
        
        self.layout["messages"].update(
            Panel(
                messages_table,
                title="[bold]Messages[/bold]",
                border_style="cyan",
                padding=(0, 1)
            )
        )
    
    def _update_status(self):
        """Update status panel"""
        status_table = Table(
            show_header=False,
            box=None,
            expand=True
        )
        
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")
        
        # Add status items
        if self.system_status:
            for key, value in self.system_status.items():
                if isinstance(value, (str, int, float, bool)):
                    status_table.add_row(key, str(value))
        else:
            status_table.add_row("Status", "Initializing...")
        
        # Add mode info
        if self.orchestrator:
            state = self.orchestrator.state_manager.get_state()
            status_table.add_row("Current Mode", state.current_mode)
            status_table.add_row("Tasks", str(len(self.tasks)))
            status_table.add_row("Files", str(len(state.file_context)))
        
        self.layout["status"].update(
            Panel(
                status_table,
                title="[bold]System Status[/bold]",
                border_style="green",
                padding=(1, 1)
            )
        )
    
    def _update_tasks(self):
        """Update tasks panel"""
        tasks_table = Table(
            show_header=True,
            header_style="bold yellow",
            box=None,
            expand=True
        )
        
        tasks_table.add_column("ID", width=8)
        tasks_table.add_column("Status", width=10)
        tasks_table.add_column("Description", ratio=1)
        
        for task in self.tasks[-10:]:  # Show last 10 tasks
            task_id = task.get("id", "N/A")
            status = task.get("status", "unknown")
            description = task.get("description", "")
            
            # Color code status
            status_colors = {
                "pending": "yellow",
                "running": "cyan",
                "completed": "green",
                "failed": "red",
                "cancelled": "dim"
            }
            
            status_style = status_colors.get(status, "white")
            status_text = Text(status.upper(), style=status_style)
            
            # Truncate description
            if len(description) > 30:
                description = description[:27] + "..."
            
            tasks_table.add_row(task_id, status_text, description)
        
        self.layout["tasks"].update(
            Panel(
                tasks_table,
                title="[bold]Active Tasks[/bold]",
                border_style="yellow",
                padding=(0, 1)
            )
        )
    
    def _update_progress(self):
        """Update progress panel"""
        if not self.progress_data:
            # Show spinner when no progress data
            spinner = SpinnerColumn()
            text = TextColumn("[progress.description]{task.description}")
            progress_bar = Progress(spinner, text, console=self.console)
            task_id = progress_bar.add_task("[cyan]Waiting for tasks...", total=None)
            
            self.layout["progress"].update(
                Panel(
                    progress_bar,
                    title="[bold]Progress[/bold]",
                    border_style="blue",
                    padding=(1, 1)
                )
            )
        else:
            # Show progress bars
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
            
            for task_name, percent in self.progress_data.items():
                progress.add_task(f"[cyan]{task_name}", total=100, completed=percent*100)
            
            self.layout["progress"].update(
                Panel(
                    progress,
                    title="[bold]Progress[/bold]",
                    border_style="blue",
                    padding=(1, 1)
                )
            )
    
    def _update_footer(self):
        """Update footer panel"""
        # Help text
        help_text = Text()
        help_text.append("Commands: ", style="bold")
        help_text.append("/help ", style="cyan")
        help_text.append("/mode ", style="green")
        help_text.append("/status ", style="yellow")
        help_text.append("/exit", style="red")
        
        # Mode info
        if self.orchestrator:
            mode_manager = self.orchestrator.mode_manager
            current_mode = mode_manager.current_mode
            
            mode_text = Text()
            mode_text.append("Current mode: ", style="dim")
            mode_text.append(current_mode.value.upper(), style="bold yellow")
            
            # Suggested next mode
            suggested = mode_manager.suggest_next_mode({})
            if suggested:
                mode_text.append(" → Next: ", style="dim")
                mode_text.append(suggested.value.upper(), style="bold green")
        
        else:
            mode_text = Text("Starting up...", style="dim")
        
        footer_content = Group(help_text, Text(""), mode_text)
        
        self.layout["footer"].update(
            Panel(
                footer_content,
                border_style="dim",
                padding=(0, 1)
            )
        )
    
    def display_message(self, message: str, sender: str = "system", msg_type: str = "message"):
        """Display a message in the UI"""
        self.post_event(UIEvent(
            type=msg_type,
            data={
                "content": message,
                "sender": sender
            }
        ))
    
    def display_error(self, error: str):
        """Display an error in the UI"""
        self.display_message(error, "system", "error")
    
    def update_system_status(self, status: Dict[str, Any]):
        """Update system status display"""
        self.post_event(UIEvent(
            type="update",
            data={"status": status}
        ))
    
    def update_tasks(self, tasks: List[Dict[str, Any]]):
        """Update tasks display"""
        self.post_event(UIEvent(
            type="update",
            data={"tasks": tasks}
        ))
    
    def update_progress(self, progress: Dict[str, float]):
        """Update progress display"""
        self.post_event(UIEvent(
            type="update",
            data={"progress": progress}
        ))
    
    def display_code(self, code: str, language: str = "python"):
        """Display code with syntax highlighting"""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Code", border_style="green"))
    
    def display_diff(self, diff: str):
        """Display a diff"""
        # Parse and colorize diff
        lines = diff.split('\n')
        colored_lines = []
        
        for line in lines:
            if line.startswith('+'):
                colored_lines.append(f"[green]{line}[/green]")
            elif line.startswith('-'):
                colored_lines.append(f"[red]{line}[/red]")
            elif line.startswith('@'):
                colored_lines.append(f"[cyan]{line}[/cyan]")
            else:
                colored_lines.append(line)
        
        diff_text = '\n'.join(colored_lines)
        self.console.print(Panel(diff_text, title="Changes", border_style="yellow"))
    
    def display_table(self, data: List[List[Any]], headers: List[str], title: str = ""):
        """Display data in a table"""
        table = Table(title=title, show_header=True, header_style="bold")
        
        for header in headers:
            table.add_column(header)
        
        for row in data:
            table.add_row(*[str(cell) for cell in row])
        
        self.console.print(table)
    
    def ask_question(self, question: str, default: Optional[str] = None) -> str:
        """Ask a question and get response"""
        return Prompt.ask(f"[cyan]{question}[/cyan]", default=default)
    
    def ask_confirmation(self, question: str, default: bool = False) -> bool:
        """Ask for confirmation"""
        return Confirm.ask(f"[yellow]{question}[/yellow]", default=default)
    
    def show_waiting(self, message: str = "Processing..."):
        """Show waiting indicator"""
        with self.console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            time.sleep(0.1)  # Placeholder
    
    def clear_screen(self):
        """Clear the screen"""
        self.console.clear()
