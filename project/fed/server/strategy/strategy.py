"""A custom strategy."""

from logging import WARNING, log

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flwr.server.strategy.aggregate import aggregate
from typing import Any


class FedDC(FedAvg):
    """Federated daisy-chaining strategy.

    Parameters
    ----------
    aggregation_period : int, optional
        The period until aggregation is performed. Defaults to 5.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(self, aggregation_period: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.aggregation_period = aggregation_period
        self.stored_parameters: list[Parameters] | None = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "FedDC"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        if (
            server_round - 1
        ) % self.aggregation_period == 0 or self.stored_parameters is None:

            # Initial parameters or aggregated parameters
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        else:
            # Distributed parameters. No shuffle is needed since the client parameters
            # in stored_parameters are randomly sampled.
            return [
                (client, FitIns(self.stored_parameters[i], config))
                for i, client in enumerate(clients)
            ]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using unweighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Save the parameters after each round of training to be redistributed.
        self.stored_parameters = [fit_res.parameters for _, fit_res in results]

        # Unweighted average
        averaged_parameters = []
        for arr in weights[0]:
            averaged_parameters.append(arr.copy())
        for weight in weights[1:]:
            for i, arr in enumerate(weight):
                averaged_parameters[i] += arr

        for i in range(len(averaged_parameters)):
            averaged_parameters[i] = averaged_parameters[i] / len(weights)

        parameters_aggregated = ndarrays_to_parameters(averaged_parameters)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


class FedDCW(FedAvg):
    """Federated daisy-chaining strategy with weighted average by samples seen.

    Parameters
    ----------
    aggregation_period : int, optional
        The period until aggregation is performed. Defaults to 5.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(self, aggregation_period: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.aggregation_period = aggregation_period
        self.stored_parameters: list[Parameters] | None = None
        self.sample_counts: list[int] | None = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "FedDCW"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        if (
            server_round - 1
        ) % self.aggregation_period == 0 or self.stored_parameters is None:
            # Reset weighted average count
            self.sample_counts = [0] * len(clients)

            # Initial parameters or aggregated parameters
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        else:
            # Distributed parameters. No shuffle is needed since the client parameters
            # in stored_parameters are randomly sampled.
            return [
                (client, FitIns(self.stored_parameters[i], config))
                for i, client in enumerate(clients)
            ]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Save the parameters after each round of training to be redistributed.
        self.stored_parameters = [fit_res.parameters for _, fit_res in results]

        # Initialise count for weighted average
        if self.sample_counts is None:
            self.sample_counts = [0] * len(results)

        for i, count in enumerate([fit_res.num_examples for _, fit_res in results]):
            self.sample_counts[i] += count

        # Weighted average
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), self.sample_counts[i])
            for i, (_, fit_res) in enumerate(results)
        ]
        averaged_parameters = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(averaged_parameters)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
