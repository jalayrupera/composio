import typing as t

from composio.tools.base.local import LocalAction, LocalTool

from .actions import (
    AdvancedRagToolQuery,
    AdvancedAddDocumentToRagTool,
    HydeQueryRewriter,
    RerankResults,
)


class AdvancedRagTool(LocalTool, autoload=True):
    """Advanced Rag Tool"""

    logo = "https://raw.githubusercontent.com/ComposioHQ/composio/master/python/docs/imgs/logos/Ragtool.png"
    slug = "ADVANCEDRAGTOOL"

    @classmethod
    def actions(cls) -> list[t.Type[LocalAction]]:
        # Add the new actions to the list
        return [
            AdvancedRagToolQuery,
            AdvancedAddDocumentToRagTool,
            HydeQueryRewriter,
            RerankResults,
        ]
