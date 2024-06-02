from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

class ArmaCell(nn.Module):
    def __init__(
        self,
        q: int,
        input_dim: Tuple[int, int],
        p: Optional[int] = None,
        units: int = 1,
        activation_name: Optional[str] = None,  
        use_bias: bool = False,
        return_lags: bool = False,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation_name
        self.activation = nn.Identity() if self.activation_name == "linear" else nn.ReLU()
        self.q = q
        self.p = p if p is not None else input_dim[1]
        self.k = input_dim[0]
        assert self.p <= input_dim[1]
        assert self.p > 0
        assert self.q > 0
        assert self.k > 0
        self.q_overhang = self.q > self.p
        self.use_bias = use_bias
        self.return_lags = return_lags

        # Set during build()
        self.kernel =  nn.Parameter(torch.Tensor(self.p, self.units, self.k, self.k))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(self.q, self.units, self.k, self.k))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.k * self.units))
        else:
            self.bias = None
        self.init_weights()

    @property
    def state_size(self) -> torch.Size:
        return torch.Size((self.k * self.units, self.q))

    @property
    def output_size(self) -> torch.Size:
        return (
            torch.Size((self.k * self.units, self.q))
            if self.return_lags
            else torch.Size((self.k * self.units, 1))
        )

    def init_weights(self):
        nn.init.uniform_(self.kernel)
        nn.init.uniform_(self.recurrent_kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input:   BATCH x k x max(p,q)
        # Output:  BATCH x (k*units) x 1      if return_lags = False
        #          BATCH x (k*units) x q      if return_lags = True

        # STATE:
        # BATCH x (k*units) x q -> BATCH x k x units x q
        input_state = states[0]
        input_state = torch.unsqueeze(input_state, dim=-2)
        input_state = torch.reshape(input_state, (-1, self.k, self.units, self.q))

        # AR
        ar_out = []
        for i in range(self.p):
            ar_out.append(inputs[:, :, i] @ self.kernel[i, :, :, :])  # type: ignore
        ar = torch.stack(ar_out, axis=-1)
        ar = torch.reshape(ar, (-1, self.k * self.units, self.p))  # BATCH x (k * units)

        # MA
        ma_out = []
        for i in range(self.q):
            ma_unit = []
            if i + 1 > self.p:
                lhs = input_state - torch.unsqueeze(inputs, dim=-2)
            else:
                lhs = input_state

            for j in range(self.units):
                ma_unit.append(lhs[:, :, j, i] @ self.recurrent_kernel[i, j, :, :])  # type: ignore
            ma_out.append(torch.stack(ma_unit, dim=-1))
        ma = sum(ma_out)
        ma = torch.reshape(ma, (-1, self.k * self.units))

        output = ar + ma

        if self.use_bias:
            output = output + self.bias

        output = self.activation(output)
        # output = torch.unsqueeze(output, axis=-1)
        output_state = torch.cat([output, states[0][:, :, :-1]], axis=-1)

        if self.return_lags:
            return output_state, output_state
        else:
            return output, output_state


class ARMA(nn.Module):
    def __init__(
        self,
        q: int,
        input_dim: Tuple[int, int],
        p: Optional[int] = None,
        units: int = 1,
        activation: str = "linear",
        use_bias: bool = False,
        return_lags: bool = False,
        return_sequences: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)  
        cell = ArmaCell(
            q, input_dim, p, units, activation, use_bias, return_lags, **kwargs
        )
        self.cell = cell
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input:   BATCH x k x max(p,q)
        # Output:  BATCH x (k*units) x 1      if return_lags = False
        #          BATCH x (k*units) x q      if return_lags = True

        # STATE:
        # BATCH x (k*units) x q -> BATCH x k x units x q
        state = torch.zeros((inputs.shape[0], self.cell.state_size[0], self.cell.state_size[1]), device=inputs.device)

        if self.cell.return_lags:
            output, _ = self.cell(inputs, (state,))
        else:
            output, _ = self.cell(inputs, (state,))
        return output

if __name__ == "__main__":
    model = ARMA(2, (1, 1))
    dummy_input = torch.randn(1, 1, 2)
    output = model(dummy_input)
    print(output.shape)
