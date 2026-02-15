"""
ASEA-X Main Application - Updated with Phase 3 features
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.core.orchestrator import Orchestrator
from src.core.state_manager import StateManager
from src.core.mode_manager import ModeManager, SystemMode
from src.core.safety_system import SafetySystem
from src.llm.deepseek_client import DeepSeekClient, LLMConfig
from src.ui.terminal_ui import TerminalUI
from src.tools.context_loader import ContextLoader
from src.tools.git_manager import GitManager
from src.execution.runtime_monitor import ExecutionObserver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="asea-x",
    help="Autonomous Software Engineering Agent System",
    add_completion=False
)


class ASEAX:
    """Enhanced ASEA-X application with Phase 3 features"""
    
    def __init__(self, workdir: Path = Path("./workdir")):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)
        
        # Initialize core components
        self.state_manager = StateManager(workdir)
        self.mode_manager = ModeManager(self.state_manager)
        self.safety_system = SafetySystem(self.state_manager)
        
        # Initialize new Phase 3 components
        self.context_loader = ContextLoader(workdir)
        self.git_manager = GitManager(workdir)
        self.execution_observer = ExecutionObserver()
        
        # Initialize LLM client
        self.llm_client = DeepSeekClient()
        
        # Initialize UI
        self.ui = TerminalUI()
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            state_manager=self.state_manager,
            mode_manager=self.mode_manager,
            safety_system=self.safety_system,
            llm_client=self.llm_client
        )
        
        # Connect UI to orchestrator
        self.ui.orchestrator = self.orchestrator
        
        self.running = False
        
    async def start(self):
        """Start the ASEA-X system with enhanced features"""
        logger.info("Starting ASEA-X system with Phase 3 features...")
        
        # Verify LLM connection
        try:
            models = self.llm_client.get_models()
            logger.info(f"Connected to LLM provider. Available models: {models}")
        except Exception as e:
            logger.error(f"Failed to connect to LLM: {e}")
            console.print("[red]❌ LLM connection failed. Check your API key and network.[/red]")
            return False
        
        # Initialize orchestrator
        await self.orchestrator.initialize()
        
        # Setup Git hooks
        if self.git_manager.is_initialized():
            self.git_manager.setup_hooks()
        
        # Start UI
        self.ui.start()
        
        self.running = True
        logger.info("ASEA-X system started successfully")
        
        # Print enhanced system info
        state = self.state_manager.get_state()
        git_status = self.git_manager.get_status()
        context_stats = self.context_loader.get_statistics()
        exec_stats = self.execution_observer.get_statistics()
        
        console.print(f"""
[bold cyan]ASEA-X System Started (Phase 3)[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[blue]Mode:[/blue] {state.current_mode}
[blue]Workdir:[/blue] {state.workdir}
[blue]Safety:[/blue] {'✅ Enabled' if state.safety_enabled else '❌ Disabled'}
[blue]LLM Model:[/blue] {self.llm_client.config.model}
[blue]Git:[/blue] {'✅ Initialized' if git_status.get('clean') is not None else '❌ Not initialized'}
[blue]Files Loaded:[/blue] {context_stats.get('total_files', 0)}
[blue]Executions:[/blue] {exec_stats.get('total_executions', 0)} ({exec_stats.get('success_rate', 0):.0f}% success)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        # Show UI instructions
        self.ui.display_message(
            "ASEA-X Phase 3 started! Features enabled:",
            "system"
        )
        self.ui.display_message(
            "• Complete Debugger Agent with auto-fixing",
            "system"
        )
        self.ui.display_message(
            "• Vector DB file context with semantic search",
            "system"
        )
        self.ui.display_message(
            "• Git integration with atomic commits",
            "system"
        )
        self.ui.display_message(
            "• Rich terminal UI with real-time updates",
            "system"
        )
        self.ui.display_message(
            "• Runtime monitoring and error analysis",
            "system"
        )
        
        return True
    
    async def stop(self):
        """Stop the ASEA-X system"""
        logger.info("Stopping ASEA-X system...")
        self.running = False
        
        # Stop UI
        self.ui.stop()
        
        # Save context
        self.context_loader.save_context()
        
        # Shutdown orchestrator
        await self.orchestrator.shutdown()
        
        logger.info("ASEA-X system stopped")
    
    async def process_command(self, command: str) -> str:
        """
        Process a user command
        
        Args:
            command: User input command
            
        Returns:
            System response
        """
        if not self.running:
            return "System not running. Start with 'asea-x start'."
        
        # Check for enhanced commands
        if command.startswith("/context"):
            return await self._handle_context_command(command)
        elif command.startswith("/git"):
            return await self._handle_git_command(command)
        elif command.startswith("/monitor"):
            return await self._handle_monitor_command(command)
        
        return await self.orchestrator.process_input(command)
    
    async def _handle_context_command(self, command: str) -> str:
        """Handle context-related commands"""
        parts = command.split()
        if len(parts) < 2:
            return "Usage: /context <load|search|stats|clear> [args]"
        
        subcommand = parts[1]
        
        if subcommand == "load":
            if len(parts) < 3:
                return "Usage: /context load <file_path|directory>"
            
            target = parts[2]
            if Path(target).is_dir():
                results = self.context_loader.load_directory(target)
                return f"Loaded {len(results)} files from {target}"
            else:
                metadata = self.context_loader.load_file(target)
                if metadata:
                    return f"Loaded {target} ({metadata.size_bytes} bytes, {metadata.line_count} lines)"
                else:
                    return f"Failed to load {target}"
        
        elif subcommand == "search":
            if len(parts) < 3:
                return "Usage: /context search <query>"
            
            query = " ".join(parts[2:])
            results = self.context_loader.search_files(query, limit=5)
            
            if not results:
                return f"No results found for: {query}"
            
            lines = [f"Search results for: {query}"]
            for i, (file_path, score, snippet) in enumerate(results, 1):
                lines.append(f"{i}. {file_path} (relevance: {score:.2f})")
                if snippet:
                    lines.append(f"   {snippet}")
            
            return "\n".join(lines)
        
        elif subcommand == "stats":
            stats = self.context_loader.get_statistics()
            lines = ["Context Statistics:"]
            lines.append(f"Total files: {stats.get('total_files', 0)}")
            lines.append(f"Total size: {stats.get('total_size_bytes', 0):,} bytes")
            lines.append("Languages:")
            for lang, count in stats.get('languages', {}).items():
                lines.append(f"  {lang}: {count} files")
            
            return "\n".join(lines)
        
        elif subcommand == "clear":
            self.context_loader.clear_context()
            return "Cleared all file context"
        
        else:
            return f"Unknown context command: {subcommand}"
    
    async def _handle_git_command(self, command: str) -> str:
        """Handle Git-related commands"""
        parts = command.split()
        if len(parts) < 2:
            return "Usage: /git <status|commit|branch|history|diff> [args]"
        
        subcommand = parts[1]
        
        if subcommand == "status":
            status = self.git_manager.get_status()
            if "error" in status:
                return f"Git error: {status['error']}"
            
            lines = ["Git Status:"]
            lines.append(f"Branch: {status.get('branch', 'unknown')}")
            lines.append(f"Clean: {'✅ Yes' if status.get('clean') else '❌ No'}")
            lines.append(f"Changed files: {len(status.get('changed_files', []))}")
            lines.append(f"Staged files: {len(status.get('staged_files', []))}")
            lines.append(f"Untracked files: {len(status.get('untracked_files', []))}")
            
            if status.get('changed_files'):
                lines.append("\nChanged files:")
                for file in status['changed_files'][:10]:
                    lines.append(f"  {file}")
            
            return "\n".join(lines)
        
        elif subcommand == "commit":
            if len(parts) < 3:
                return "Usage: /git commit <message> [--type <type>] [--scope <scope>]"
            
            message = " ".join(parts[2:])
            
            # Parse optional flags
            commit_type = "feat"
            scope = None
            
            if "--type" in parts:
                idx = parts.index("--type")
                if idx + 1 < len(parts):
                    commit_type = parts[idx + 1]
            
            if "--scope" in parts:
                idx = parts.index("--scope")
                if idx + 1 < len(parts):
                    scope = parts[idx + 1]
            
            commit_hash = self.git_manager.commit_changes(
                message=message,
                commit_type=commit_type,
                scope=scope
            )
            
            if commit_hash:
                return f"Committed changes: {commit_hash}"
            else:
                return "No changes to commit or commit failed"
        
        elif subcommand == "branch":
            branches = self.git_manager.get_branches()
            if not branches:
                return "No branches found"
            
            lines = ["Branches:"]
            for branch in branches:
                current = " * " if branch.is_current else "   "
                lines.append(f"{current}{branch.name}")
                if branch.last_commit:
                    lines.append(f"      Last commit: {branch.last_commit.message}")
            
            return "\n".join(lines)
        
        elif subcommand == "history":
            limit = 10
            if len(parts) > 2:
                try:
                    limit = int(parts[2])
                except:
                    pass
            
            commits = self.git_manager.get_commit_history(limit=limit)
            if not commits:
                return "No commit history"
            
            lines = [f"Last {len(commits)} commits:"]
            for commit in commits:
                lines.append(f"{commit.hash[:8]} {commit.date.strftime('%Y-%m-%d')} {commit.message}")
            
            return "\n".join(lines)
        
        elif subcommand == "diff":
            diff = self.git_manager.get_diff()
            if not diff:
                return "No changes to diff"
            
            # Show diff in UI
            self.ui.display_diff(diff)
            return "Showing diff in UI"
        
        else:
            return f"Unknown Git command: {subcommand}"
    
    async def _handle_monitor_command(self, command: str) -> str:
        """Handle monitor-related commands"""
        parts = command.split()
        if len(parts) < 2:
            return "Usage: /monitor <stats|history|errors|clear>"
        
        subcommand = parts[1]
        
        if subcommand == "stats":
            stats = self.execution_observer.get_statistics()
            lines = ["Execution Monitor Statistics:"]
            lines.append(f"Total executions: {stats['total_executions']}")
            lines.append(f"Successful: {stats['successful']}")
            lines.append(f"Failed: {stats['failed']}")
            lines.append(f"Success rate: {stats['success_rate']:.1f}%")
            lines.append(f"Average time: {stats['average_time']:.2f}s")
            lines.append(f"Total time: {stats['total_time']:.1f}s")
            
            if stats['common_errors']:
                lines.append("\nCommon errors:")
                for error, count in stats['common_errors'].items():
                    lines.append(f"  {error}: {count} times")
            
            return "\n".join(lines)
        
        elif subcommand == "history":
            limit = 5
            if len(parts) > 2:
                try:
                    limit = int(parts[2])
                except:
                    pass
            
            executions = self.execution_observer.get_recent_executions(limit=limit)
            if not executions:
                return "No execution history"
            
            lines = [f"Recent executions (last {len(executions)}):"]
            for exec_result in executions:
                status_icon = "✅" if exec_result.state == "completed" else "❌"
                lines.append(f"{status_icon} {exec_result.command[:50]}...")
                lines.append(f"   State: {exec_result.state}, Time: {exec_result.execution_time:.2f}s")
                if exec_result.error_type:
                    lines.append(f"   Error: {exec_result.error_type}")
            
            return "\n".join(lines)
        
        elif subcommand == "errors":
            analysis = self.execution_observer.get_error_analysis()
            if "message" in analysis:
                return analysis["message"]
            
            lines = ["Error Analysis:"]
            for error_type, details in analysis.items():
                lines.append(f"\n{error_type.upper()}:")
                lines.append(f"  Count: {details['count']}")
                lines.append(f"  Frequency: {details['frequency']:.1%}")
                lines.append("  Solutions:")
                for solution in details.get('common_solutions', []):
                    lines.append(f"    • {solution}")
            
            return "\n".join(lines)
        
        elif subcommand == "clear":
            self.execution_observer.clear_history()
            return "Cleared execution history"
        
        else:
            return f"Unknown monitor command: {subcommand}"
    
    async def interactive_session(self):
        """Start interactive session with enhanced UI"""
        # UI is already running, just wait for exit
        console.print("[bold cyan]ASEA-X Interactive Mode[/bold cyan]")
        console.print("UI is running. Use the terminal interface for commands.")
        console.print("Type 'exit' in the UI or press Ctrl+C here to quit.\n")
        
        while self.running:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                break


@app.command()
def start(
    workdir: Path = typer.Option(
        "./workdir",
        "--workdir", "-w",
        help="Working directory for ASEA-X"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive", "-i",
        help="Start interactive session"
    ),
    ui: bool = typer.Option(
        True,
        "--ui", "-u",
        help="Enable rich terminal UI"
    )
):
    """Start ASEA-X system with enhanced features"""
    asea_x = ASEAX(workdir)

    # Start system
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Setup signal handlers (schedule stop on the running loop)
    def signal_handler(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        # Schedule coroutine safely onto the existing loop
        asyncio.run_coroutine_threadsafe(asea_x.stop(), loop)
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start system
        success = loop.run_until_complete(asea_x.start())
        if not success:
            console.print("[red]Failed to start ASEA-X[/red]")
            sys.exit(1)
        
        # Start interactive session if requested
        if interactive:
            if ui:
                # UI runs in its own thread, just wait
                console.print("[green]✅ Rich UI started. Switch to UI window for interaction.[/green]")
                console.print("[yellow]⚠️  This terminal is now for monitoring only.[/yellow]")
            loop.run_until_complete(asea_x.interactive_session())
        else:
            # Keep running for API mode
            console.print("[yellow]Running in API mode. Press Ctrl+C to exit.[/yellow]")
            loop.run_forever()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        loop.run_until_complete(asea_x.stop())
        loop.close()


@app.command()
def demo():
    """Run ASEA-X demo with example workflow"""
    console.print("[bold cyan]ASEA-X Demo Mode[/bold cyan]")
    console.print("This will demonstrate the complete ASEA-X workflow.\n")
    
    # Create demo directory
    demo_dir = Path("./demo_project")
    demo_dir.mkdir(exist_ok=True)
    
    # Initialize ASEA-X
    asea_x = ASEAX(demo_dir)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        console.print("\n[yellow]Demo interrupted[/yellow]")
        asyncio.run(asea_x.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run demo
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start system
        success = loop.run_until_complete(asea_x.start())
        if not success:
            console.print("[red]Failed to start demo[/red]")
            return
        
        console.print("\n[bold green]Demo Steps:[/bold green]")
        console.print("1. Planning a Python web API")
        console.print("2. Writing the API code")
        console.print("3. Running linting checks")
        console.print("4. Testing the implementation")
        console.print("5. Git integration demo\n")
        
        # Demo steps
        import time
        
        console.print("[cyan]Step 1: Planning[/cyan]")
        result = loop.run_until_complete(
            asea_x.orchestrator.process_input("/planner")
        )
        console.print(result)
        
        time.sleep(1)
        
        console.print("\n[cyan]Step 2: Development[/cyan]")
        result = loop.run_until_complete(
            asea_x.orchestrator.process_input("/dev")
        )
        console.print(result)
        
        time.sleep(1)
        
        console.print("\n[cyan]Step 3: Git Integration[/cyan]")
        result = loop.run_until_complete(
            asea_x.process_input("/git status")
        )
        console.print(result)
        
        console.print("\n[green]✅ Demo completed successfully![/green]")
        console.print("Run 'asea-x start --interactive' for full interactive mode.")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
    finally:
        loop.run_until_complete(asea_x.stop())
        loop.close()


# Keep existing commands: version, reset, status
@app.command()
def version():
    """Show ASEA-X version"""
    from src import __version__
    console.print(f"[bold cyan]ASEA-X[/bold cyan] version {__version__} (Phase 3 Complete)")


@app.command()
def reset(
    workdir: Path = typer.Option(
        "./workdir",
        "--workdir", "-w",
        help="Working directory to reset"
    ),
    confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation"
    )
):
    """Reset ASEA-X workdir"""
    if not confirm:
        typer.confirm(
            f"Are you sure you want to reset {workdir}? This will delete all agent work.",
            abort=True
        )
    
    import shutil
    
    if workdir.exists():
        shutil.rmtree(workdir)
        console.print(f"[green]Reset {workdir}[/green]")
    else:
        console.print(f"[yellow]{workdir} does not exist[/yellow]")


@app.command()
def status(
    workdir: Path = typer.Option(
        "./workdir",
        "--workdir", "-w",
        help="Working directory to check"
    )
):
    """Check ASEA-X status"""
    if not workdir.exists():
        console.print(f"[red]Workdir {workdir} does not exist[/red]")
        return
    
    # Check for system state
    state_file = workdir / "system_state.json"
    if not state_file.exists():
        console.print(f"[yellow]No system state found in {workdir}[/yellow]")
        return
    
    import json
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Check for context
    context_file = workdir / "file_context.json"
    has_context = context_file.exists()
    
    # Check for Git
    git_dir = workdir / ".git"
    has_git = git_dir.exists()
    
    console.print(f"""
[bold cyan]ASEA-X Status (Phase 3)[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[blue]Workdir:[/blue] {workdir}
[blue]Mode:[/blue] {state.get('current_mode', 'unknown')}
[blue]Tasks:[/blue] {len(state.get('task_history', []))}
[blue]Safety:[/blue] {'✅ Enabled' if state.get('safety_enabled', True) else '❌ Disabled'}
[blue]Files in context:[/blue] {len(state.get('file_context', {}))}
[blue]Vector DB:[/blue] {'✅ Loaded' if has_context else '❌ Not loaded'}
[blue]Git:[/blue] {'✅ Initialized' if has_git else '❌ Not initialized'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    app()
