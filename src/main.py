"""
ASEA-X Main Application
Autonomous Software Engineering Agent System
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
    """Main ASEA-X application class"""
    
    def __init__(self, workdir: Path = Path("./workdir")):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)
        
        # Initialize core components
        self.state_manager = StateManager(workdir)
        self.mode_manager = ModeManager(self.state_manager)
        self.safety_system = SafetySystem(self.state_manager)
        
        # Initialize LLM client
        self.llm_client = DeepSeekClient()
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            state_manager=self.state_manager,
            mode_manager=self.mode_manager,
            safety_system=self.safety_system,
            llm_client=self.llm_client
        )
        
        self.running = False
        
    async def start(self):
        """Start the ASEA-X system"""
        logger.info("Starting ASEA-X system...")
        
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
        
        self.running = True
        logger.info("ASEA-X system started successfully")
        
        # Print system info
        state = self.state_manager.get_state()
        console.print(f"""
[bold cyan]ASEA-X System Started[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[blue]Mode:[/blue] {state.current_mode}
[blue]Workdir:[/blue] {state.workdir}
[blue]Safety:[/blue] {'✅ Enabled' if state.safety_enabled else '❌ Disabled'}
[blue]LLM Model:[/blue] {self.llm_client.config.model}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        return True
    
    async def stop(self):
        """Stop the ASEA-X system"""
        logger.info("Stopping ASEA-X system...")
        self.running = False
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
        
        return await self.orchestrator.process_input(command)
    
    async def interactive_session(self):
        """Start interactive session"""
        console.print("[bold cyan]ASEA-X Interactive Mode[/bold cyan]")
        console.print("Type '/help' for commands, '/exit' to quit\n")
        
        while self.running:
            try:
                # Get user input
                prompt = f"[bold green]{self.state_manager.get_state().current_mode}[/bold green] > "
                user_input = await asyncio.to_thread(
                    console.input,
                    prompt
                )
                
                # Check for exit
                if user_input.strip().lower() in ['/exit', '/quit', 'exit', 'quit']:
                    break
                
                # Process command
                response = await self.process_command(user_input)
                
                # Display response
                console.print(f"\n[blue]System:[/blue] {response}\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Use /exit to quit.[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                console.print(f"[red]Error: {e}[/red]")
        
        await self.stop()


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
    )
):
    """Start ASEA-X system"""
    asea_x = ASEAX(workdir)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        asyncio.run(asea_x.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start system
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start system
        success = loop.run_until_complete(asea_x.start())
        if not success:
            console.print("[red]Failed to start ASEA-X[/red]")
            sys.exit(1)
        
        # Start interactive session if requested
        if interactive:
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
def version():
    """Show ASEA-X version"""
    from src import __version__
    console.print(f"[bold cyan]ASEA-X[/bold cyan] version {__version__}")


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
    
    state_file = workdir / "system_state.json"
    if not state_file.exists():
        console.print(f"[yellow]No system state found in {workdir}[/yellow]")
        return
    
    import json
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    console.print(f"""
[bold cyan]ASEA-X Status[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━
[blue]Workdir:[/blue] {workdir}
[blue]Mode:[/blue] {state.get('current_mode', 'unknown')}
[blue]Tasks:[/blue] {len(state.get('task_history', []))}
[blue]Safety:[/blue] {'✅ Enabled' if state.get('safety_enabled', True) else '❌ Disabled'}
[blue]Files in context:[/blue] {len(state.get('file_context', {}))}
━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    app()
