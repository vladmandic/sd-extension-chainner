import logging
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import install

log = logging.getLogger("sd")
logger = log
console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
    "traceback.border": "black",
    "traceback.border.syntax_error": "black",
    "inspect.value.border": "black",
}))
install(console=console)

class PB(Progress):
    def __init__(self):
        super().__init__(TextColumn('[cyan]{task.description}'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), TimeElapsedColumn(), console=console)


if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.DEBUG, console=console)
    rh.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s | %(module)s | %(message)s', handlers=[rh])
    log.debug('Initialized chaiNNer logging')
