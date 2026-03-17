"""Optional local-only server scaffold for oLLM."""

from ollm.server.runtime import (
    SERVER_EXTRA_INSTALL_HINT as SERVER_EXTRA_INSTALL_HINT,
)
from ollm.server.runtime import (
    ServerDependenciesError as ServerDependenciesError,
)
from ollm.server.runtime import (
    create_server_app as create_server_app,
)
from ollm.server.runtime import (
    serve_application as serve_application,
)
