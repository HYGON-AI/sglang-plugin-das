# SPDX-License-Identifier: Apache-2.0
import socket
from typing import Any, Dict, Optional

import torch
from sglang.srt.weight_cache import transport as upstream
from sglang.srt.weight_cache.transport import MultiprocessingSerializer, send_msg


class TorchIpcTransportBackend(upstream.TorchIpcTransportBackend):
    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "preloaded_weights_bytes": preloaded_weights_bytes,
                "extra_state": extra_state,
            },
        )

    def export_extra_tensor(self, tensor: torch.Tensor) -> Any:
        return MultiprocessingSerializer.serialize(tensor.data, output_str=True)

    def import_extra_tensor(self, handle: Any) -> torch.Tensor:
        return MultiprocessingSerializer.deserialize(handle)



def get_client_transport_backend(name):
    if name in (None, "", upstream.TORCH_IPC_BACKEND):
        return TorchIpcTransportBackend()
    return upstream.get_client_transport_backend(name)
