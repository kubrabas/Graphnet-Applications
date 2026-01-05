#######################################################################################
# Model 


class Model(
    Logger, Configurable, LightningModule, ABC, metaclass=ModelConfigSaverABC
):
    """Base class for all components in graphnet."""

    verbose_print = True

    @staticmethod
    def _get_batch_size(data: List[Data]) -> int:
        return sum([torch.numel(torch.unique(d.batch)) for d in data])

    def save(self, path: str) -> None:
        """Save entire model to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix for model files."
            )
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.cpu(), path, pickle_module=dill)
        self.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load entire model from `path`."""
        return torch.load(path, pickle_module=dill)

    def save_state_dict(self, path: str) -> None:
        """Save model `state_dict` to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix "
                "for state_dict files."
            )
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)
        self.info(f"Model state_dict saved to {path}")

    def load_state_dict(  # type: ignore[override]
        self, path: Union[str, Dict], **kargs: Optional[Any]
    ) -> "Model":  # pylint: disable=arguments-differ
        """Load model `state_dict` from `path`."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path

        # DEPRECATION UTILITY: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647
        state_dict, state_dict_altered = rename_state_dict_entries(
            state_dict=state_dict, old_phrase="_gnn", new_phrase="backbone"
        )
        if state_dict_altered:
            self.warning(
                "DeprecationWarning: State dicts with `_gnn`"
                " entries will be deprecated in GraphNeT 2.0"
            )
        return super().load_state_dict(state_dict, **kargs)

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[ModelConfig, str],
        trust: bool = False,
        load_modules: Optional[List[str]] = None,
    ) -> "Model":
        """Construct `Model` instance from `source` configuration.

        Arguments:
            trust: Whether to trust the ModelConfig file enough to `eval(...)`
                any lambda function expressions contained.
            load_modules: List of modules used in the definition of the model
                which, as a consequence, need to be loaded into the global
                namespace. Defaults to loading `torch`.

        Raises:
            ValueError: If the ModelConfig contains lambda functions but
                `trust = False`.
        """
        if isinstance(source, str):
            source = ModelConfig.load(source)

        assert isinstance(
            source, ModelConfig
        ), f"Argument `source` of type ({type(source)}) is not a `ModelConfig"

        return source._construct_model(trust, load_modules)

    def set_verbose_print_recursively(self, verbose_print: bool) -> None:
        """Set verbose_print recursively for all Model modules."""
        for module in self.modules():
            if isinstance(module, Model):
                module.verbose_print = verbose_print
        self.verbose_print = verbose_print

    def extra_repr(self) -> str:
        """Provide a more detailed description of the object print.

        Returns:
            str: A string representation containing detailed information
            about the object.
        """
        return self._extra_repr() if self.verbose_print else ""

    def _extra_repr(self) -> str:
        """Detailed information about the object."""
        return f"""{self.__class__.__name__}(\n{self.extra_repr_recursive(
            self._config.__dict__)})"""

    def extra_repr_recursive(self, dictionary: dict, indent: int = 4) -> str:
        """Recursively format a dictionary for extra_repr."""
        result = "{\n"
        for key, value in dictionary.items():
            if key == "class_name":
                continue
            result += " " * indent + f"'{key}': "
            if isinstance(value, dict):
                result += self.extra_repr_recursive(value, indent + 4)
            elif isinstance(value, Model):
                result += value.__repr__()
            else:
                result += repr(value)
            result += ",\n"
        result += " " * (indent - 4) + "}"
        return result
    




#######################################################################################
# Detector 


class Detector(Model):
    """Base class for all detector-specific read-ins in graphnet."""

    def __init__(
        self, replace_with_identity: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`.

        Args:
            replace_with_identity: A list of feature names from the
            feature_map that should be replaced with the identity
            function.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._replace_with_identity = replace_with_identity

    @abstractmethod
    def feature_map(self) -> Dict[str, Callable]:
        """List of features used/assumed by inheriting `Detector` objects."""

    @final
    def forward(  # type: ignore
        self, input_features: torch.tensor, input_feature_names: List[str]
    ) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""
        return self._standardize(input_features, input_feature_names)

    @property
    def geometry_table(self) -> pd.DataFrame:
        """Public get method for retrieving a `Detector`s geometry table."""
        if ~hasattr(self, "_geometry_table"):
            try:
                assert hasattr(self, "geometry_table_path")
            except AssertionError as e:
                self.error(
                    f"""{self.__class__.__name__} does not have class
                           variable `geometry_table_path` set."""
                )
                raise e
            self._geometry_table = pd.read_parquet(self.geometry_table_path)
        return self._geometry_table

    @property
    def string_index_name(self) -> str:
        """Public get method for retrieving the string index column name."""
        return self.string_id_column

    @property
    def sensor_position_names(self) -> List[str]:
        """Public get method for retrieving the xyz coordinate column names."""
        return self.xyz

    @property
    def sensor_index_name(self) -> str:
        """Public get method for retrieving the sensor id column name."""
        return self.sensor_id_column

    @final
    def _standardize(
        self, input_features: torch.tensor, input_feature_names: List[str]
    ) -> Data:
        feature_map = self.feature_map()
        if self._replace_with_identity is not None:
            for feature in self._replace_with_identity:
                feature_map[feature] = self._identity
        for idx, feature in enumerate(input_feature_names):
            try:
                input_features[:, idx] = feature_map[
                    feature
                ](  # noqa: E501 # type: ignore
                    input_features[:, idx]
                )
            except KeyError as e:
                self.warning(
                    f"""No Standardization function found for '{feature}'"""
                )
                raise e
        return input_features

    def _identity(self, x: torch.tensor) -> torch.tensor:
        """Apply no standardization to input."""
        return x
    

#######################################################################################
# PONE 

class PONE(Detector):
    """Detector class for P-ONE."""

    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        return {
            "dom_x": self._xyz,
            "dom_y": self._xyz,
            "dom_z": self._xyz,
            "dom_time": self._time,
            "charge": self._charge,
        }

    def _xyz(self, x: torch.Tensor) -> torch.Tensor:
        return x  # x / 500.0

    def _time(self, x: torch.Tensor) -> torch.Tensor:
        return x  # (x - 1.0e4) / 3.0e4

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return x  # torch.log10(x)



#######################################################################################
#  NodeDefinition


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def __init__(
        self, input_feature_names: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        if input_feature_names is not None:
            self.set_output_feature_names(
                input_feature_names=input_feature_names
            )

    @final
    def forward(self, x: torch.tensor) -> torch.tensor:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
        """
        data = self._construct_nodes(x=x)

        return data

    @property
    def _output_feature_names(self) -> List[str]:
        """Return output feature names."""
        try:
            self._hidden_output_feature_names
        except AttributeError as e:
            self.error(
                f"""{self.__class__.__name__} was instantiated without
                       `input_feature_names` and it was not set prior to this
                       forward call. If you are using this class outside a
                       `GraphDefinition`, please instatiate
                       with `input_feature_names`."""
            )  # noqa
            raise e
        return self._hidden_output_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting
        classes.
        """
        return len(self._output_feature_names)

    @final
    def set_number_of_inputs(self, input_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            input_feature_names: name of each input feature column.
        """
        assert isinstance(input_feature_names, list)
        self.nb_inputs = len(input_feature_names)

    @final
    def set_output_feature_names(self, input_feature_names: List[str]) -> None:
        """Set output features names as a member variable.

        Args:
            input_feature_names: List of column names of the input to the
            node definition.
        """
        self._hidden_output_feature_names = self._define_output_feature_names(
            input_feature_names
        )

    @abstractmethod
    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Construct names of output columns.

        Args:
            input_feature_names: List of column names for the input data.

        Returns:
            A list of column names for each column in
            the node definition output.
        """

    @abstractmethod
    def _construct_nodes(self, x: torch.tensor) -> torch.tensor:
        """Construct nodes from raw node features ´x´.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            feature_names: List of names for reach column in `x`. Identical
            order of appearance. Length `d`.

        Returns:
            graph: graph without edges.
            new_node_features: A list of node features names.
        """


#######################################################################################
#  ClusterSummaryFeatures


class ClusterSummaryFeatures(NodeDefinition):
    """Represent pulse maps as clusters with summary features.

    If `cluster_on` is set to the xyz coordinates of optical modules
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each node will be
    a unique optical module and the pulse information (e.g. charge, time)
    is summarized.
    NOTE: Developed to be used with features
        [dom_x, dom_y, dom_z, charge, time]

    Possible features per cluster:
    - total charge
        feature name: `total_charge`
    - charge accumulated after <X> time units
        feature name: `charge_after_<X>ns`
    - time of first hit in the optical module
        feature name: `time_of_first_hit`
    - time spread per optical module
        feature name: `time_spread`
    - time std per optical module
        feature name: `time_std`
    - time took to collect <X> percent of total charge per cluster
        feature name: `time_after_charge_pct<X>`
    - number of pulses per clusters
        feature name: `counts`

    For more details on some of the features see
    Theo Glauchs thesis (chapter 5.3):
    https://mediatum.ub.tum.de/node?id=1584755
    """

    def __init__(
        self,
        cluster_on: List[str],
        input_feature_names: List[str],
        charge_label: str = "charge",
        time_label: str = "dom_time",
        total_charge: bool = True,
        charge_after_t: List[int] = [10, 50, 100],
        time_of_first_hit: bool = True,
        time_spread: bool = True,
        time_std: bool = True,
        time_after_charge_pct: List[int] = [1, 3, 5, 11, 15, 20, 50, 80],
        charge_standardization: Union[float, str] = "log",
        time_standardization: float = 1e-3,
        order_in_time: bool = True,
        add_counts: bool = False,
    ) -> None:
        """Construct `ClusterSummaryFeatures`.

        Args:
            cluster_on: Names of features to create clusters from.
            input_feature_names: Column names for input features.
            charge_label: Name of the charge column.
            time_label: Name of the time column.
            total_charge: If True, calculates total charge as feature.
            charge_after_t: List of times at which the accumulated charge
                is calculated as a feature.
            time_of_first_hit: If True, time of first hit is added
                as a feature.
            time_spread: If True, time spread is added as a feature.
            time_std: If True, time std is added as a feature.
            time_after_charge_pct: List of percentiles to calculate time after
                charge.
            charge_standardization: Either a float or 'log'. If a float,
                the features are multiplied by this factor. If 'log', the
                features are transformed to log10 scale.
            time_standardization: Standardization factor for features
                with a time
            order_in_time: If True, clusters are ordered in time.
                    If your data is already ordered in time, you can set this
                    to False to avoid a potential overhead.
                NOTE: Should only be set to False if you are sure that
                    the input data is already ordered in time. Will lead to
                    incorrect results otherwise.
            add_counts: If True, number of log10(event counts per clusters)
                is added as a feature.

        NOTE: Make sure that either the input data is not already standardized
        or that the `charge_standardization` and `time_standardization`
        parameters are set to 1 to avoid a double standardization.
        """
        # Set member variables
        self._cluster_on = cluster_on
        self._charge_label = charge_label
        self._time_label = time_label
        self._order_in_time = order_in_time

        # Check if charge_standardization is a float or 'log'
        self._charge_standardization = charge_standardization
        self._time_standardization = time_standardization
        self._verify_standardization()

        # feature member variables
        self._total_charge = total_charge
        self._charge_after_t = charge_after_t
        self._time_of_first_hit = time_of_first_hit
        self._time_spread = time_spread
        self._time_std = time_std
        self._time_after_charge_pct = time_after_charge_pct
        self._add_counts = add_counts

        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)
        if self._order_in_time is False:
            self.info(
                "Setting `order_by_time` to False. "
                "Make sure that the input data is already ordered in time."
            )

    def _define_output_feature_names(
        self,
        input_feature_names: List[str],
    ) -> List[str]:
        """Set the output feature names."""
        self.set_indices(input_feature_names)
        new_feature_names = deepcopy(self._cluster_on)
        if self._total_charge:
            new_feature_names.append("total_charge")
        for t in self._charge_after_t:
            new_feature_names.append(f"charge_after_{t}ns")
        if self._time_of_first_hit:
            new_feature_names.append("time_of_first_hit")
        if self._time_spread:
            new_feature_names.append("time_spread")
        if self._time_std:
            new_feature_names.append("time_std")
        for pct in self._time_after_charge_pct:
            new_feature_names.append(f"time_after_charge_pct{pct}")
        if self._add_counts:
            new_feature_names.append("counts")
        return new_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        cluster_class = cluster_and_pad(
            x=x,
            cluster_columns=self._cluster_idx,
            sort_by=[self._time_idx] if self._order_in_time else [],
        )
        # calculate charge weighted median time as reference
        ref_time = cluster_class.reference_time(
            charge_index=self._charge_idx,
            time_index=self._time_idx,
        )

        # add total charge
        if self._total_charge:
            cluster_class.add_sum_charge(charge_index=self._charge_idx)
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._charge_standardization,
            )

        # add charge after t
        if len(self._charge_after_t) > 0:
            cluster_class.add_accumulated_value_after_t(
                time_index=self._time_idx,
                summarization_indices=[self._charge_idx],
                times=self._charge_after_t,
            )
            cluster_class.clustered_x[:, -len(self._charge_after_t) :] = (
                self._standardize_features(
                    cluster_class.clustered_x[:, -len(self._charge_after_t) :],
                    self._charge_standardization,
                )
            )

        # add time of first hit
        if self._time_of_first_hit:
            cluster_class.add_time_first_pulse(
                time_index=self._time_idx,
            )
            cluster_class.clustered_x[:, -1] -= ref_time

            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time spread
        if self._time_spread:
            cluster_class.add_spread(
                columns=[self._time_idx],
            )
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time std
        if self._time_std:
            cluster_class.add_std(
                columns=[self._time_idx],
            )
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time after charge percentiles
        if len(self._time_after_charge_pct) > 0:
            cluster_class.add_charge_threshold_summary(
                summarization_indices=[self._time_idx],
                percentiles=self._time_after_charge_pct,
                charge_index=self._charge_idx,
            )
            cluster_class.clustered_x[
                :, -len(self._time_after_charge_pct) :
            ] -= ref_time
            cluster_class.clustered_x[
                :, -len(self._time_after_charge_pct) :
            ] = self._standardize_features(
                cluster_class.clustered_x[
                    :, -len(self._time_after_charge_pct) :
                ],
                self._time_standardization,
            )

        if self._add_counts:
            cluster_class.add_counts()
        return torch.tensor(cluster_class.clustered_x)

    def set_indices(self, feature_names: List[str]) -> None:
        """Set the indices for the input features."""
        self._cluster_idx = [
            feature_names.index(column) for column in self._cluster_on
        ]
        self._charge_idx = feature_names.index(self._charge_label)
        self._time_idx = feature_names.index(self._time_label)

    def _standardize_features(
        self,
        x: np.ndarray,
        standardization: Union[float, str],
    ) -> np.ndarray:
        """Standardize the features in the input array."""
        if isinstance(standardization, float):
            return x * standardization
        elif standardization == "log":
            return np.log10(x)
        else:
            # should never happen, but just in case
            raise ValueError(
                f"standardization must be either a float or 'log', "
                f"but got {standardization}"
            )

    def _verify_standardization(
        self,
    ) -> torch.Tensor:
        """Verify settings of standardization of the features."""
        if not isinstance(self._charge_standardization, float):
            if isinstance(self._charge_standardization, str):
                if self._charge_standardization != "log":
                    raise ValueError(
                        f"charge_standardization must be either a float or"
                        f" 'log', but got {self._charge_standardization}"
                    )
            else:
                raise ValueError(
                    f"charge_standardization must be either a float or 'log', "
                    f"but got {self._charge_standardization}"
                )

        if not isinstance(self._time_standardization, float):
            raise ValueError(
                f"time_standardization must be a float, "
                f"but got {self._time_standardization}"
            )
        



#######################################################################################
#  DataRepresentation


class DataRepresentation(Model):
    """An Abstract class to create data representations from."""

    def __init__(
        self,
        detector: Detector,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
        repeat_labels: bool = False,
    ):
        """Construct´DataRepresentation´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            input_feature_names: Names of each column in expected input data
                that will be built into a the data. If not provided,
                it is automatically assumed that all features in `Detector` is
                used.
            dtype: data type used for features. e.g. ´torch.float´
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
            add_inactive_sensors: If True, inactive sensors will be appended
                to the data with padded pulse information. Defaults to False.
            sensor_mask: A list of sensor id's to be masked from the data. Any
                sensor listed here will be removed from the data.
                    Defaults to None.
            string_mask: A list of string id's to be masked from the data.
                Defaults to None.
            repeat_labels: If True, labels will be repeated to match the
                the number of rows in the output of the GraphDefinition.
                Defaults to False.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member Variables
        self._detector = detector
        self._perturbation_dict = perturbation_dict
        self._sensor_mask = sensor_mask
        self._string_mask = string_mask
        self._add_inactive_sensors = add_inactive_sensors
        self._repeat_labels = repeat_labels

        self._resolve_masks()

        if input_feature_names is None:
            # Assume all features in Detector is used.
            input_feature_names = list(
                self._detector.feature_map().keys()
            )  # noqa: E501 # type: ignore
        self._input_feature_names = input_feature_names

        self.nb_inputs = len(self._input_feature_names)

        # Set data type
        self.to(dtype)

        # Set perturbation_cols if needed
        if isinstance(self._perturbation_dict, dict):
            self._perturbation_cols = [
                self._input_feature_names.index(key)
                for key in self._perturbation_dict.keys()
            ]
        if seed is not None:
            if isinstance(seed, int):
                self.rng = default_rng(seed)
            elif isinstance(seed, Generator):
                self.rng = seed
            else:
                raise ValueError(
                    "Invalid seed. Must be an int or a numpy Generator."
                )
        else:
            self.rng = default_rng()

    @property
    def nb_outputs(self) -> int:
        """Return the number of output features."""
        return len(self.output_feature_names)

    @property
    def output_feature_names(self) -> List[str]:
        """Initialize / return the names of output features."""
        if not hasattr(self, "_output_feature_names"):
            self._output_feature_names = self._set_output_feature_names(
                self._input_feature_names
            )
        return self._output_feature_names

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Construct data as ´Data´ object.

        Args:
            input_features: Input features for data construction.
                Shape ´[num_rows, d]´
            input_feature_names: name of each column. Shape ´[,d]´.
            truth_dicts: Dictionary containing truth labels.
            custom_label_functions: Custom label functions.
            loss_weight_column: Name of column that holds loss weight.
                                Defaults to None.
            loss_weight: Loss weight associated with event. Defaults to None.
            loss_weight_default_value: default value for loss weight.
                    Used in instances where some events have
                    no pre-defined loss weight. Defaults to None.
            data_path: Path to dataset data files. Defaults to None.

        Returns:
            data
        """
        # Checks
        self._validate_input(
            input_features=input_features,
            input_feature_names=input_feature_names,
        )

        # Add inactive sensors if `add_inactive_sensors = True`
        if self._add_inactive_sensors:
            input_features = self._attach_inactive_sensors(
                input_features, input_feature_names
            )

        # Mask out sensors if `sensor_mask` is given
        if self._sensor_mask is not None:
            input_features = self._mask_sensors(
                input_features, input_feature_names
            )

        # Gaussian perturbation of each column if perturbation dict is given
        input_features = self._perturb_input(input_features)

        # Transform to pytorch tensor
        input_features = torch.tensor(input_features, dtype=self.dtype)

        # Standardize / Scale final data features
        input_features = self._detector(input_features, input_feature_names)

        # Create data & get new final data feature names
        data = Data(x=input_features)

        # Attach number of pulses as static attribute.
        data.n_pulses = torch.tensor(len(input_features), dtype=torch.int32)

        # Attach data path - useful for Ensemble datasets.
        if data_path is not None:
            data["dataset_path"] = data_path

        # Attach loss weights if they exist
        data = self._add_loss_weights(
            data=data,
            loss_weight=loss_weight,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
        )

        # Attach default truth labels and node truths
        if truth_dicts is not None:
            data = self._add_truth(data=data, truth_dicts=truth_dicts)

        # Attach custom truth labels
        if custom_label_functions is not None:
            data = self._add_custom_labels(
                data=data, custom_label_functions=custom_label_functions
            )

        # DEPRECATION STAMP GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647
        data["graph_definition"] = self.__class__.__name__

        # Add data representation Stamp
        data["data_representation"] = self.__class__.__name__
        return data

    def _resolve_masks(self) -> None:
        """Handle cases with sensor/string masks."""
        if self._sensor_mask is not None:
            if self._string_mask is not None:
                assert (
                    1 == 2
                ), "Please specify only one of `sensor_mask`and `string_mask`."

        if (self._sensor_mask is None) & (self._string_mask is not None):
            self._sensor_mask = self._convert_string_to_sensor_mask()

        return

    def _convert_string_to_sensor_mask(self) -> List[int]:
        """Convert a string mask to a sensor mask."""
        string_id_column = self._detector.string_id_column
        sensor_id_column = self._detector.sensor_id_column
        geometry_table = self._detector.geometry_table
        idx = geometry_table[string_id_column].isin(self._string_mask)
        return np.asarray(geometry_table.loc[idx, sensor_id_column]).tolist()

    def _attach_inactive_sensors(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Attach inactive sensors to `input_features`.

        This function will query the detector geometry table and add any
        sensor in the geometry table that is not already present in
        queried data per event.
        """
        lookup = self._geometry_table_lookup(
            input_features, input_feature_names
        )
        geometry_table = self._detector.geometry_table
        unique_sensors = geometry_table.reset_index(drop=True)

        # multiple lines to avoid long line:
        inactive_idx = ~geometry_table.index.isin(lookup)
        inactive_sensors = unique_sensors.loc[
            inactive_idx, input_feature_names
        ]
        input_features = np.concatenate(
            [input_features, inactive_sensors.to_numpy()], axis=0
        )
        return input_features

    def _mask_sensors(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Mask sensors according to `sensor_mask`."""
        sensor_id_column = self._detector.sensor_index_name
        geometry_table = self._detector.geometry_table

        lookup = self._geometry_table_lookup(
            input_features=input_features,
            input_feature_names=input_feature_names,
        )
        mask = ~geometry_table.loc[lookup, sensor_id_column].isin(
            self._sensor_mask
        )

        return input_features[mask, :]

    def _geometry_table_lookup(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Convert xyz in `input_features` into a set of sensor ids."""
        lookup_columns = [
            input_feature_names.index(feature)
            for feature in self._detector.sensor_position_names
        ]
        idx = [*zip(*[tuple(input_features[:, k]) for k in lookup_columns])]
        return self._detector.geometry_table.loc[idx, :].index

    def _validate_input(
        self, input_features: np.array, input_feature_names: List[str]
    ) -> None:
        # raw data feature matrix dimension check
        assert input_features.shape[1] == len(input_feature_names)

        # check that provided features for input is the same that the
        # `DataRepresentation` was instantiated with.
        assert len(input_feature_names) == len(
            self._input_feature_names
        ), f"""Input features ({input_feature_names}) is not what
               {self.__class__.__name__} was instatiated
               with ({self._input_feature_names})"""  # noqa
        for idx in range(len(input_feature_names)):
            assert (
                input_feature_names[idx] == self._input_feature_names[idx]
            ), f""" Order of representation features in data
                    are not the same as expected. Got {input_feature_names}
                    vs. {self._input_feature_names}"""  # noqa

    def _perturb_input(self, input_features: np.ndarray) -> np.ndarray:
        if isinstance(self._perturbation_dict, dict):
            self.warning_once(
                f"""Will randomly perturb
                {list(self._perturbation_dict.keys())}
                using stds {self._perturbation_dict.values()}"""  # noqa
            )
            perturbed_features = self.rng.normal(
                loc=input_features[:, self._perturbation_cols],
                scale=np.array(
                    list(self._perturbation_dict.values()), dtype=float
                ),
            )
            input_features[:, self._perturbation_cols] = perturbed_features
        return input_features

    def _add_loss_weights(
        self,
        data: Data,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
    ) -> Data:
        """Attempt to store a loss weight in the data for use during training.

        I.e. `data[loss_weight_column] = loss_weight`

        Args:
            loss_weight: The non-negative weight to be stored.
            data: Data object representing the event.
            loss_weight_column: The name under which the weight is stored in
                                 the data.
            loss_weight_default_value: The default value used if
                                        none was retrieved.

        Returns:
            A data with loss weight added, if available.
        """
        # Add loss weight to data.
        if loss_weight is not None and loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                data[loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self.dtype
                ).reshape(-1, 1)
            else:
                data[loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self.dtype
                ).reshape(-1, 1)
        return data

    def _add_custom_labels(
        self,
        data: Data,
        custom_label_functions: Dict[str, Callable[..., Any]],
    ) -> Data:
        # Add custom labels to the data
        for key, fn in custom_label_functions.items():
            label = fn(data)
            if self._repeat_labels:
                label = self._label_repeater(label, data)
            data[key] = label
        return data

    def _add_truth(
        self, data: Data, truth_dicts: List[Dict[str, Any]]
    ) -> Data:
        """Add truth labels from ´truth_dicts´ to ´data´.

        I.e. ´data[key] = truth_dict[key]´


        Args:
            data: data where the label will be stored
            truth_dicts: dictionary containing the labels

        Returns:
            data with labels
        """
        # Write attributes, either target labels, truth info or original
        # features.

        for truth_dict in truth_dicts:
            for key, value in truth_dict.items():
                try:
                    label = torch.tensor(value)
                    if self._repeat_labels:
                        label = self._label_repeater(label, data)
                    data[key] = label
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to data."
                        )
                    )
        return data

    @abstractmethod
    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Set the final data output feature names."""
        raise NotImplementedError

    def _label_repeater(self, label: torch.Tensor, data: Data) -> torch.Tensor:
        """Handle the label repetition.

        Necessary only if the `repeat_labels` argument is being used
        E.g: for graphs: `label.repeat(data.x.shape[0], 1)`
        """
        raise NotImplementedError


#######################################################################################
#  GraphDefinition


class GraphDefinition(DataRepresentation):
    """An Abstract class to create graph definitions from."""

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        edge_definition: Optional[EdgeDefinition] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
        sort_by: Optional[str] = None,
        repeat_labels: bool = False,
        add_static_features: bool = True,
    ):
        """Construct ´GraphDefinition´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        ´node_definition´ defines the nodes in the graph.

        ´edge_definition´ defines the connectivity of the nodes in the graph.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            node_definition: Definition of nodes. Defaults to NodesAsPulses.
            edge_definition: Definition of edges. Defaults to None.
            input_feature_names: Names of each column in expected input data
                that will be built into a graph. If not provided,
                it is automatically assumed that all features in `Detector` is
                used.
            dtype: data type used for node features. e.g. ´torch.float´
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
            add_inactive_sensors: If True, inactive sensors will be appended
                to the graph with padded pulse information. Defaults to False.
            sensor_mask: A list of sensor id's to be masked from the graph. Any
                sensor listed here will be removed from the graph.
                    Defaults to None.
            string_mask: A list of string id's to be masked from the graph.
                Defaults to None.
            sort_by: Name of node feature to sort by. Defaults to None.
            repeat_labels: If True, labels will be repeated to match the
                the number of rows in the output of the GraphDefinition.
                Defaults to False.
            add_static_features: If True, the original features will be
                added as static attributes to the graph. Defaults to True.
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            input_feature_names=input_feature_names,
            dtype=dtype,
            perturbation_dict=perturbation_dict,
            seed=seed,
            add_inactive_sensors=add_inactive_sensors,
            sensor_mask=sensor_mask,
            string_mask=string_mask,
            repeat_labels=repeat_labels,
        )

        if node_definition is None:
            node_definition = NodesAsPulses()
        self._node_definition = node_definition
        self._edge_definition = edge_definition
        if self._edge_definition is None:
            self.warning_once(
                """No EdgeDefinition given. Graphs will not have edges!"""
            )

        # Sorting
        if sort_by is not None:
            assert isinstance(sort_by, str)
            try:
                sort_by = self.output_feature_names.index(  # type: ignore
                    sort_by
                )  # type: ignore
            except ValueError as e:
                self.error(
                    f"{sort_by} not in node "
                    f"features {self.output_feature_names}."
                )
                raise e
        self._sort_by = sort_by
        self._add_static_features = add_static_features

        # make sure output feature names are set also in node definition
        self._set_output_feature_names(self._input_feature_names)

    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Set the final data output feature names."""
        # Set input data column names for node definition
        self._node_definition.set_output_feature_names(input_feature_names)
        return self._node_definition._output_feature_names

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Construct graph as ´Data´ object.

        Args:
            input_features: Input features for graph construction.
                Shape ´[num_rows, d]´
            input_feature_names: name of each column. Shape ´[,d]´.
            truth_dicts: Dictionary containing truth labels.
            custom_label_functions: Custom label functions.
            loss_weight_column: Name of column that holds loss weight.
                                Defaults to None.
            loss_weight: Loss weight associated with event. Defaults to None.
            loss_weight_default_value: default value for loss weight.
                    Used in instances where some events have
                    no pre-defined loss weight. Defaults to None.
            data_path: Path to dataset data files. Defaults to None.

        Returns:
            graph
        """
        data = super().forward(
            input_features=input_features,
            input_feature_names=input_feature_names,
            truth_dicts=truth_dicts,
            custom_label_functions=custom_label_functions,
            loss_weight_column=loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=loss_weight_default_value,
            data_path=data_path,
        )
        # Create graph & get new node feature names
        data.x = self._node_definition(data.x)
        if self._sort_by is not None:
            data.x = data.x[data.x[:, self._sort_by].sort()[1]]

        # Enforce dtype
        data.x = data.x.type(self.dtype)

        # Assign edges
        if self._edge_definition is not None:
            data = self._edge_definition(data)

        if self._add_static_features:
            data = self._add_features_individually(
                data,
                self.output_feature_names,
            )
        return data

    def _add_features_individually(
        self,
        data: Data,
        data_feature_names: List[str],
    ) -> Data:
        # Additionally add original features as (static) attributes
        data.features = data_feature_names
        for index, feature in enumerate(data_feature_names):
            if feature not in ["x"]:  # reserved for graph features.
                data[feature] = data.x[:, index].detach()
            else:
                self.warning_once(
                    """Cannot assign data['x']. This field is reserved for
                      graph node features. Please rename your input feature."""
                )  # noqa

        return data

    def _label_repeater(self, label: torch.Tensor, data: Data) -> torch.Tensor:
        """Handle the label repetition.

        Not a classmethod because it is not necessary for all data
        representations.
        """
        return label.repeat(data.x.shape[0], 1)
    



#######################################################################################
#  KNNGraph


class KNNGraph(GraphDefinition):
    """A Graph representation where Edges are drawn to nearest neighbours."""

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
        distance_as_edge_feature: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            input_feature_names: Name of input feature columns.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                Defaults to None.
            nb_nearest_neighbours: Number of edges for each node.
                Defaults to 8.
            columns: node feature columns used for distance calculation.
                Defaults to [0, 1, 2].
            distance_as_edge_feature: Add edge distances as an edge feature.
                Defaults to False.
        """
        # Base class constructor
        edge_definition = (
            KNNDistanceEdges if distance_as_edge_feature else KNNEdges
        )
        super().__init__(
            detector=detector,
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=edge_definition(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )


#######################################################################################
#  Dataset


"""Base :py:class:`Dataset` class(es) used in GraphNeT."""

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterable,
    Type,
)

import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.utilities.string_selection_resolver import (
    StringSelectionResolver,
)
from graphnet.training.labels import Label
from graphnet.utilities.config import (
    Configurable,
    DatasetConfig,
    DatasetConfigSaverABCMeta,
)
from graphnet.exceptions.exceptions import ColumnMissingException
from graphnet.utilities.logging import Logger
from graphnet.models.data_representation import (
    GraphDefinition,
    DataRepresentation,
)
from graphnet.utilities.config.parsing import (
    get_all_grapnet_classes,
)


def load_module(class_name: str) -> Type:
    return namespace_classes[class_name]


def parse_data_representation(data_rep_cfg: dict) -> DataRepresentation:
    return data_representation


def parse_labels(cfg: dict) -> Dict[str, Label]:
    return labels


class Dataset(
    Logger,
    Configurable,
    torch.utils.data.Dataset,
    ABC,
    metaclass=DatasetConfigSaverABCMeta,
):
    """Base Dataset class for reading from any intermediate file format."""

    # Class method(s)
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[DatasetConfig, str],
    ) -> Union[
        "Dataset",
        "EnsembleDataset",
        Dict[str, "Dataset"],
        Dict[str, "EnsembleDataset"],
    ]:
        """Construct `Dataset` instance from `source` configuration."""
        if isinstance(source, str):
            source = DatasetConfig.load(source)

        assert isinstance(source, DatasetConfig), (
            f"Argument `source` of type ({type(source)}) is not a "
            "`DatasetConfig`"
        )

        if "data_representation" not in source.dict().keys():
            if "graph_definition" in source.dict().keys():
                Logger(log_folder=None).warning_once(
                    "DeprecationWarning: Field `graph_definition` will be"
                    " deprecated in GraphNeT 2.0. Please use "
                    "`data_representation` instead."
                )
            else:
                raise TypeError(
                    "`DatasetConfig` incompatible with "
                    "current GraphNeT version."
                )

        # Parse set of `selection``.
        if isinstance(source.selection, dict):
            return cls._construct_datasets_from_dict(source)
        elif (
            isinstance(source.selection, list)
            and len(source.selection)
            and isinstance(source.selection[0], str)
        ):
            return cls._construct_dataset_from_list_of_strings(source)

        cfg = source.dict()

        if (
            "data_representation" in cfg
            and cfg["data_representation"] is not None
        ):
            cfg["data_representation"] = parse_data_representation(
                cfg["data_representation"]
            )
        elif "graph_definition" in cfg and cfg["graph_definition"] is not None:
            Logger(log_folder=None).warning_once(
                "DeprecationWarning: Field `graph_definition` will be"
                " deprecated in GraphNeT 2.0. Please use "
                "`data_representation` instead."
            )
            cfg["graph_definition"] = parse_data_representation(
                cfg["graph_definition"]
            )

        if cfg["labels"] is not None:
            cfg["labels"] = parse_labels(cfg)

        if isinstance(cfg["path"], list):
            sources = []
            for path in cfg["path"]:
                cfg["path"] = path
                sources.append(source._dataset_class(**cfg))
            source = EnsembleDataset(sources)
            return source
        else:
            return source._dataset_class(**cfg)

    @classmethod
    def concatenate(
        cls,
        datasets: List["Dataset"],
    ) -> "EnsembleDataset":
        """Concatenate multiple `Dataset`s into one instance."""
        return EnsembleDataset(datasets)

    @classmethod
    def _construct_datasets_from_dict(
        cls, config: DatasetConfig
    ) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in dict `self.selection`."""
        assert isinstance(config.selection, dict)
        datasets: Dict[str, "Dataset"] = {}
        selections: Dict[str, Union[str, List]] = deepcopy(config.selection)
        for key, selection in selections.items():
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, (Dataset, EnsembleDataset))
            datasets[key] = dataset

        # Reset `selections`.
        config.selection = selections

        return datasets

    @classmethod
    def _construct_dataset_from_list_of_strings(
        cls, config: DatasetConfig
    ) -> "Dataset":
        """Construct `Dataset` for each entry in list `self.selection`."""
        assert isinstance(config.selection, list)
        datasets: List["Dataset"] = []
        selections: List[str] = deepcopy(cast(List[str], config.selection))
        for selection in selections:
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            datasets.append(dataset)

        # Reset `selections`.
        config.selection = selections

        return cls.concatenate(datasets)

    @classmethod
    def _resolve_graphnet_paths(
        cls, path: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        if isinstance(path, list):
            return [cast(str, cls._resolve_graphnet_paths(p)) for p in path]

        assert isinstance(path, str)
        return (
            path.replace("$graphnet", GRAPHNET_ROOT_DIR)
            .replace("$GRAPHNET", GRAPHNET_ROOT_DIR)
            .replace("${graphnet}", GRAPHNET_ROOT_DIR)
            .replace("${GRAPHNET}", GRAPHNET_ROOT_DIR)
        )

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        graph_definition: Optional[GraphDefinition] = None,
        data_representation: Optional[DataRepresentation] = None,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        labels: Optional[Dict[str, Any]] = None,
    ):
        """Construct Dataset.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The events that should be read. This can be given either
                as list of indicies (in `index_column`); or a string-based
                selection used to query the `Dataset` for events passing the
                selection. Defaults to None, meaning that all events in the
                input files are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
            data_representation: Method that defines the data representation.
            labels: Dictionary of labels to be added to the dataset.

            graph_definition: Method that defines the graph representation.
                NOTE: DEPRECATED Use `data_representation` instead.
                # DEPRECATION: REMOVE AT 2.0 LAUNCH
                # See https://github.com/graphnet-team/graphnet/issues/647
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Resolve reference to `$GRAPHNET` in path(s)
        path = self._resolve_graphnet_paths(path)

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value

        if data_representation is None:
            if graph_definition is not None:
                data_representation = graph_definition
                # Code continues after warning
                self.warning(
                    "DeprecationWarning: Argument `graph_definition` "
                    "will be deprecated in GraphNeT 2.0. "
                    "Please use `data_representation` instead."
                )
            else:
                # Code stops
                raise TypeError(
                    "__init__() missing 1 required keyword argument:"
                    "'data_representation'"
                )

        self._data_representation = deepcopy(data_representation)
        self._labels = labels
        self._string_column = data_representation._detector.string_index_name

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.warning(
                (
                    "String selection detected.\n "
                    f"Accepted strings: {string_selection}\n "
                    "All other strings are ignored!"
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        self._selection = None
        if self._string_selection:
            # Broken into multple lines lines for length
            col = self._string_column
            condition = str(tuple(self._string_selection))
            self._selection = f"{col} in {condition}"

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        self._label_fns: Dict[str, Callable[[Data], Any]] = {}

        self._string_selection_resolver = StringSelectionResolver(
            self,
            index_column=index_column,
            seed=seed,
        )

        if self._labels is not None:
            for key in self._labels.keys():
                self.add_label(self._labels[key])

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        self._indices: Union[List[int], List[List[int]]]
        if selection is None:
            self._indices = self._get_all_indices()
        elif isinstance(selection, str):
            self._indices = self._resolve_string_selection_to_indices(
                selection
            )
        else:
            self._indices = selection

        # Implementation-specific post-init code.
        self._post_init()

    # Properties
    @property
    def path(self) -> Union[str, List[str]]:
        """Path to the file(s) from which this `Dataset` reads."""
        return self._path

    @property
    def truth_table(self) -> str:
        """Name of the table containing event-level truth information."""
        return self._truth_table

    # DEPRECATION PROPERTY: REMOVE AT 2.0 LAUNCH
    # See https://github.com/graphnet-team/graphnet/issues/647
    @property
    def _graph_definition(self) -> DataRepresentation:
        """Return the graph definition."""
        self.warning(
            "DeprecationWarning: `_graph_definition` will be deprecated in"
            " GraphNeT 2.0. Please use `_data_representation` instead."
        )
        return self._data_representation

    # Abstract method(s)
    @abstractmethod
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""

    def _post_init(self) -> None:
        """Implementation-specific code executed after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all unique values in `self._index_column`."""

    @abstractmethod
    def _get_event_index(self, sequential_index: int) -> int:
        """Return the event index corresponding to a `sequential_index`."""

    @abstractmethod
    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def add_label(
        self, fn: Callable[[Data], Any], key: Optional[str] = None
    ) -> None:
        """Add custom graph label define using function `fn`."""
        if isinstance(fn, Label):
            key = fn.key
        assert isinstance(
            key, str
        ), "Please specify a key for the custom label to be added."
        assert (
            key not in self._label_fns
        ), f"A custom label {key} has already been defined."
        self._label_fns[key] = fn

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._indices)

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    # Internal method(s)
    def _resolve_string_selection_to_indices(
        self, selection: str
    ) -> List[int]:
        """Resolve selection as string to list of indices.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        return self._string_selection_resolver.resolve(selection)

    def _remove_missing_columns(self) -> None:
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Check if table is completely empty
        if len(self) == 0:
            self.warning("Dataset is empty.")
            return

        # Find missing features
        missing_features_set = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features_set = missing_features_set.intersection(missing)

        missing_features = list(missing_features_set)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self.query_table(
                    table=table, columns=[column], sequential_index=0
                )
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)
            except IndexError:
                self.warning(f"Dataset contains no entries for {column}")

        return self._missing_variables.get(table, [])

    def _query(
        self, sequential_index: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """Query file for event features and truth information.

        The returned lists have lengths corresponding to the number of pulses
        in the event. Their constituent tuples have lengths corresponding to
        the number of features/attributes in each output

        Args:
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """
        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self.query_table(
                pulsemap, self._features, sequential_index, self._selection
            )
            features.append(features_pulsemap)

        if len(self._pulsemaps) > 0:
            features = np.concatenate(features, axis=0)

        truth = self.query_table(
            self._truth_table, self._truth, sequential_index
        )
        if self._node_truth:
            assert self._node_truth_table is not None
            node_truth = self.query_table(
                self._node_truth_table,
                self._node_truth,
                sequential_index,
                self._selection,
            )
        else:
            node_truth = None

        if self._loss_weight_column is not None:
            assert self._loss_weight_table is not None
            loss_weight = self.query_table(
                self._loss_weight_table,
                self._loss_weight_column,
                sequential_index,
            )
        else:
            loss_weight = None
        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: np.ndarray,
        truth: np.ndarray,
        node_truth: Optional[np.ndarray] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert truth to dict
        if len(truth.shape) == 1:
            truth = truth.reshape(1, -1)
        truth_dict = {
            key: truth[:, index] for index, key in enumerate(self._truth)
        }

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Create list of truth dicts with labels
        truth_dicts = [labels_dict, truth_dict]
        if node_truth is not None:
            truth_dicts.append(node_truth_dict)

        # Catch cases with no reconstructed pulses
        if len(features):
            node_features = features
        else:
            node_features = np.array([]).reshape((0, len(self._features)))

        assert isinstance(features, np.ndarray)
        # Construct graph data object
        assert self._data_representation is not None
        graph = self._data_representation(
            input_features=node_features,
            input_feature_names=self._features,
            truth_dicts=truth_dicts,
            custom_label_functions=self._label_fns,
            loss_weight_column=self._loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=self._loss_weight_default_value,
            data_path=self._path,
        )
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        if "pid" in truth_dict.keys():
            abs_pid = abs(truth_dict["pid"])

            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": int(abs_pid == 13),
                "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
                "neutrino": int(
                    (abs_pid != 13) & (abs_pid != 1)
                ),  # @TODO: `abs_pid in [12,14,16]`?
                "v_e": int(abs_pid == 12),
                "v_u": int(abs_pid == 14),
                "v_t": int(abs_pid == 16),
                "track": int(
                    (abs_pid == 14) & (truth_dict.get("interaction_type") == 1)
                ),
                "dbang": self._get_dbang_label(truth_dict),
                "corsika": int(abs_pid > 20),
            }
        else:
            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": -1,
                "muon_stopped": -1,
                "noise": -1,
                "neutrino": -1,
                "v_e": -1,
                "v_u": -1,
                "v_t": -1,
                "track": -1,
                "dbang": -1,
                "corsika": -1,
            }
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1



#######################################################################################
#  ParquetDataset

"""Base :py:class:`Dataset` class(es) used in GraphNeT."""

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Any,
)

import numpy as np
import torch
import os
from torch_geometric.data import Data
import polars as pol
from polars.series.series import Series
from glob import glob
from bisect import bisect_right
from collections import OrderedDict

from graphnet.models.data_representation import (
    GraphDefinition,
    DataRepresentation,
)
from graphnet.data.dataset import Dataset
from graphnet.exceptions.exceptions import ColumnMissingException


class ParquetDataset(Dataset):
    """Dataset class for Parquet-files converted with `ParquetWriter`."""

    def __init__(
        self,
        path: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        data_representation: Optional[DataRepresentation] = None,
        graph_definition: Optional[GraphDefinition] = None,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        cache_size: int = 1,
        labels: Optional[Dict[str, Any]] = None,
    ):
        """Construct Dataset.

            NOTE: DataLoaders using this Dataset should have
            "multiprocessing_context = 'spawn'" set to avoid thread locking.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The batch ids to include in the dataset.
                        Defaults to None, meaning that batches are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
            data_representation: Method that defines the data representation.
            graph_definition: Method that defines the graph representation.
                NOTE: DEPRECATED Use `data_representation` instead.
            cache_size: Number of files to cache in memory.
                        Must be at least 1. Defaults to 1.
            labels: Dictionary of labels to be added to the dataset.
        """
        self._validate_selection(selection)
        # Base class constructor
        super().__init__(
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            node_truth=node_truth,
            index_column=index_column,
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            string_selection=string_selection,
            selection=selection,
            dtype=dtype,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            seed=seed,
            graph_definition=graph_definition,
            data_representation=data_representation,
            labels=labels,
        )

        # mypy..
        assert isinstance(self._path, str)
        self._path: str = self._path
        # Member Variables
        self._cache_size = cache_size
        self._chunk_sizes = self._calculate_sizes()
        self._chunk_cumsum = np.cumsum(self._chunk_sizes)
        self._file_cache = self._initialize_file_cache(
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            pulsemaps=pulsemaps,
        )
        self._string_selection = string_selection
        # Purely internal member variables
        self._missing_variables: Dict[str, List[str]] = {}
        self._remove_missing_columns()

    def _initialize_file_cache(
        self,
        truth_table: str,
        node_truth_table: Optional[str],
        pulsemaps: Union[str, List[str]],
    ) -> Dict[str, OrderedDict]:
        tables = [truth_table]
        if node_truth_table is not None:
            tables.append(node_truth_table)
        if isinstance(pulsemaps, str):
            tables.append(pulsemaps)
        elif isinstance(pulsemaps, list):
            tables.extend(pulsemaps)

        cache: Dict[str, OrderedDict] = {}
        for table in tables:
            cache[table] = OrderedDict()
        return cache

    def _validate_selection(
        self,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
    ) -> None:
        if selection is not None:
            try:
                assert not isinstance(selection, str)
            except AssertionError:
                e = AssertionError(
                    f"{self.__class__.__name__} does not support "
                    "str-selections."
                )
                raise e

    def _init(self) -> None:
        return

    def _get_event_index(self, sequential_index: int) -> int:
        event_index = self.query_table(
            table=self._truth_table,
            sequential_index=sequential_index,
            columns=[self._index_column],
        )
        return event_index

    @property
    def chunk_sizes(self) -> List[int]:
        """Return a list of the chunk sizes."""
        return self._chunk_sizes

    def __len__(self) -> int:
        """Return length of dataset, i.e. number of training examples."""
        return sum(self._chunk_sizes)

    def _get_all_indices(self) -> List[int]:
        """Return a list of all unique values in `self._index_column`."""
        files = glob(os.path.join(self._path, self._truth_table, "*.parquet"))
        return np.arange(0, len(files), 1)

    def _calculate_sizes(self) -> List[int]:
        """Calculate the number of events in each chunk."""
        sizes = []
        for chunk_id in self._indices:
            path = os.path.join(
                self._path,
                self._truth_table,
                f"{self.truth_table}_{chunk_id}.parquet",
            )
            sizes.append(len(pol.read_parquet(path)))
        return sizes

    def _get_row_idx(self, sequential_index: int) -> int:
        """Return the row index corresponding to a `sequential_index`."""
        file_idx = bisect_right(self._chunk_cumsum, sequential_index)
        if file_idx > 0:
            idx = int(sequential_index - self._chunk_cumsum[file_idx - 1])
        else:
            idx = sequential_index
        return idx

    def query_table(  # type: ignore
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """
        if isinstance(columns, str):
            columns = [columns]

        if sequential_index is None:
            file_idx = np.arange(0, len(self._chunk_cumsum), 1)
        else:
            file_idx = [bisect_right(self._chunk_cumsum, sequential_index)]

        file_indices = [self._indices[idx] for idx in file_idx]

        arrays = []
        for file_idx in file_indices:
            array = self._query_table(
                table=table,
                columns=columns,
                file_idx=file_idx,
                sequential_index=sequential_index,
                selection=selection,
            )
            arrays.append(array)
        return np.concatenate(arrays, axis=0)

    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        file_idx: int,
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:

        self._load_table(table_name=table, file_idx=file_idx)
        df = self._file_cache[table][file_idx]
        if sequential_index is not None:
            row_id = self._get_row_idx(sequential_index)
        else:
            row_id = np.arange(0, len(df), 1)
        df = df[row_id]
        if len(df) > 0:
            self._raise_column_exception(
                df_columns=df.columns, columns=columns, table=table
            )
            data = df.select(columns)
            if isinstance(data[columns[0]][0], Series):
                x = [data[col][0].to_numpy().reshape(-1, 1) for col in columns]
                array = np.concatenate(x, axis=1)
            else:
                array = data.to_numpy()
        else:
            array = np.array()
        return array

    def _load_table(self, table_name: str, file_idx: int) -> None:
        """Load and possibly cache a parquet table."""
        if file_idx not in self._file_cache[table_name].keys():
            file_path = os.path.join(
                self._path, table_name, f"{table_name}_{file_idx}.parquet"
            )
            df = pol.read_parquet(file_path).sort(self._index_column)
            if (table_name in self._pulsemaps) or (
                table_name == self._node_truth_table
            ):
                if table_name == self._node_truth_table:
                    pol_columns = [pol.col(self._node_truth)]
                else:
                    pol_columns = [pol.col(feat) for feat in self._features]

                if self._string_selection:
                    pol_columns.append(pol.col(self._string_column))

                df = df.group_by(self._index_column).agg(pol_columns)

            self._file_cache[table_name][file_idx] = df.sort(
                self._index_column
            )
            n_files_cached: int = len(self._file_cache[table_name])
            if n_files_cached > self._cache_size:
                del self._file_cache[table_name][
                    list(self._file_cache[table_name].keys())[0]
                ]

    def _raise_column_exception(
        self, df_columns: List[str], columns: Union[List[str], str], table: str
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            if column not in df_columns:
                raise ColumnMissingException(f"{column} not in {table}")

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        if self._node_truth_table is not None:
            assert isinstance(self._node_truth, (list, str))  # mypy..
            node_truth = self.query_table(
                table=self._node_truth_table,
                columns=self._node_truth,
                sequential_index=sequential_index,
            )
        else:
            node_truth = None

        if self._loss_weight_table is not None:
            assert isinstance(self._loss_weight_column, str)
            loss_weight = self.query_table(
                table=self._loss_weight_table,
                columns=self._loss_weight_column,
                sequential_index=sequential_index,
            )
        else:
            loss_weight = None

        features = []
        for pulsemap in self._pulsemaps:
            features.append(
                self.query_table(
                    table=pulsemap,
                    columns=self._features,
                    sequential_index=sequential_index,
                )
            )
        features = np.concatenate(features, axis=0)

        truth = self.query_table(
            table=self._truth_table,
            columns=self._truth,
            sequential_index=sequential_index,
        )

        graph = self._create_graph(
            features=features,
            truth=truth,
            node_truth=node_truth,
            loss_weight=loss_weight,
        )
        return graph

#######################################################################################
#  DataLoader

"""Base `Dataloader` class(es) used in `graphnet`."""

from typing import Any, Callable, Dict, List, Union

import torch.utils.data
from torch_geometric.data import Batch, Data

from graphnet.data.dataset import Dataset
from graphnet.utilities.config import DatasetConfig


def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


def do_shuffle(selection_name: str) -> bool:
    """Check whether to shuffle selection with name `selection_name`."""
    return "train" in selection_name.lower()


class DataLoader(torch.utils.data.DataLoader):
    """Class for loading data from a `Dataset`."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        persistent_workers: bool = True,
        collate_fn: Callable = collate_fn,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        """Construct `DataLoader`."""
        # Base class constructor
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            **kwargs,
        )

    @classmethod
    def from_dataset_config(
        cls,
        config: DatasetConfig,
        **kwargs: Any,
    ) -> Union["DataLoader", Dict[str, "DataLoader"]]:
        """Construct `DataLoader`s based on selections in `DatasetConfig`."""
        if isinstance(config.selection, dict):
            assert "shuffle" not in kwargs, (
                "When passing a `DatasetConfig` with multiple selections, "
                "`shuffle` is automatically inferred from the selection name, "
                "and thus should not specified as an argument."
            )
            datasets = Dataset.from_config(config)
            assert isinstance(datasets, dict)
            data_loaders: Dict[str, DataLoader] = {}
            for name, dataset in datasets.items():
                data_loaders[name] = cls(
                    dataset,
                    shuffle=do_shuffle(name),
                    **kwargs,
                )

            return data_loaders

        else:
            assert "shuffle" in kwargs, (
                "When passing a `DatasetConfig` with a single selections, you "
                "need to specify `shuffle` as an argument."
            )
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            return cls(dataset, **kwargs)

#######################################################################################
#  GNN


"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import Model


class GNN(Model):
    """Base class for all core GNN models in graphnet."""

    def __init__(self, nb_inputs: int, nb_outputs: int) -> None:
        """Construct `GNN`."""
        # Base class constructor
        super().__init__()

        # Member variables
        self._nb_inputs = nb_inputs
        self._nb_outputs = nb_outputs

    @property
    def nb_inputs(self) -> int:
        """Return number of input features."""
        return self._nb_inputs

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return self._nb_outputs

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass in model."""


#######################################################################################
#  DynEdgeConv

"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union, List

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import (
    knn_graph,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from torch.nn.functional import linear
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch, softmax
from torch_scatter import scatter

from pytorch_lightning import LightningModule
from torch_geometric.utils import degree


class DynEdgeConv(EdgeConv, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(self.device)

        return x, edge_index



#######################################################################################
#  DynEdge


"""Implementation of the DynEdge GNN model architecture."""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdge(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
        activation_layer: Optional[str] = None,
        add_norm_layer: bool = False,
        skip_readout: bool = False,
    ):
        """Construct `DynEdge`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
            activation_layer: The activation function to use in the model.
            add_norm_layer: Whether to add a normalization layer after each
                linear layer.
            skip_readout: Whether to skip the readout layer(s). If `True`, the
                output of the last post-processing layer is returned directly.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        if activation_layer is None or activation_layer.lower() == "relu":
            activation_layer = torch.nn.ReLU()
        elif activation_layer.lower() == "gelu":
            activation_layer = torch.nn.GELU()
        else:
            raise ValueError(
                f"Activation layer {activation_layer} not supported."
            )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = activation_layer
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        self._add_norm_layer = add_norm_layer
        self._skip_readout = skip_readout

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                if self._add_norm_layer:
                    layers.append(torch.nn.LayerNorm(nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing operations
        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
            + nb_input_features
        )

        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(
            self._post_processing_layer_sizes
        )
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            if self._add_norm_layer:
                post_processing_layers.append(torch.nn.LayerNorm(nb_out))
            post_processing_layers.append(self._activation)

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(data.n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2)
                * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)

        # DynEdge-convolutions
        skip_connections = [x]
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        # Skip-cat
        x = torch.cat(skip_connections, dim=1)

        # Post-processing
        x = self._post_processing(x)

        if not self._skip_readout:
            # (Optional) Global pooling
            if self._global_pooling_schemes:
                x = self._global_pooling(x, batch=batch)
                if self._add_global_variables_after_pooling:
                    x = torch.cat(
                        [
                            x,
                            global_variables,
                        ],
                        dim=1,
                    )

            # Read-out
            x = self._readout(x)

        return x



#######################################################################################
#  LossFunction


"""Collection of loss functions.

All loss functions inherit from `LossFunction` which ensures a common syntax,
handles per-event weights, etc.
"""

from abc import abstractmethod
from typing import Any, Optional, Union, List, Dict

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import (
    one_hot,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    softplus,
)

from graphnet.models.model import Model
from graphnet.utilities.decorators import final


class LossFunction(Model):
    """Base class for loss functions in `graphnet`."""

    def __init__(self, **kwargs: Any) -> None:
        """Construct `LossFunction`, saving model config."""
        super().__init__(**kwargs)

    @final
    def forward(  # type: ignore[override]
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.

        Args:
            prediction: Tensor containing predictions. Shape [N,P]
            target: Tensor containing targets. Shape [N,T]
            return_elements: Whether elementwise loss terms should be returned.
                The alternative is to return the averaged loss across examples.

        Returns:
            Loss, either averaged to a scalar (if `return_elements = False`) or
            elementwise terms with shape [N,] (if `return_elements = True`).
        """
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        assert elements.size(dim=0) == target.size(
            dim=0
        ), "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax like `.forward`, for implentation in inheriting classes."""




#######################################################################################
#  StandardLearnedTask


"""Base physics task-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Any, TYPE_CHECKING, List, Tuple, Union
from typing import Callable, Optional
import numpy as np
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import Data

if TYPE_CHECKING:
    # Avoid cyclic dependency
    from graphnet.training.loss_functions import (
        LossFunction,
    )  # noqa: E501 # type: ignore[attr-defined]

from graphnet.models import Model
from graphnet.utilities.decorators import final
from graphnet.models.utils import get_fields
from graphnet.utilities.imports import has_jammy_flows_package

if has_jammy_flows_package():
    import jammy_flows


class Task(Model):
    """Base class for Tasks in GraphNeT."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @property
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    def __init__(
        self,
        *,
        target_labels: Optional[Union[str, List[str]]] = None,
        prediction_labels: Optional[Union[str, List[str]]] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        transform_support: Optional[Tuple] = None,
        loss_weight: Optional[str] = None,
    ):
        """Construct `Task`.

        Args:
            target_labels: Name(s) of the quantity/-ies being predicted, used
                to extract the  target tensor(s) from the `Data` object in
                `.compute_loss(...)`.
            prediction_labels: The name(s) of each column that is predicted by
                the model during inference. If not given, the name will auto
                matically be set to `target_label + _pred`.
            transform_prediction_and_target: Optional function to transform
                both the predicted and target tensor before passing them to the
                loss function. Useful e.g. for having the model predict
                quantities on a physical scale, but transforming this scale to
                O(1) for a numerically stable loss computation.
            transform_target: Optional function to transform only the target
                tensor before passing it, and the predicted tensor, to the loss
                function. Useful e.g. for having the model predict a
                transformed version of the target quantity, e.g. the log10-
                scaled energy, rather than the physical quantity itself. Used
                in conjunction with `transform_inference` to perform the
                inverse transform on the predicted quantity to recover the
                physical scale.
            transform_inference: Optional function to inverse-transform the
                model prediction to recover a physical scale. Used in
                conjunction with `transform_target`.
            transform_support: Optional tuple to specify minimum and maximum
                of the range of validity for the inverse transforms
                `transform_target` and `transform_inference` in case this is
                restricted. By default the invertibility of `transform_target`
                is tested on the range [-1e6, 1e6].
            loss_weight: Name of the attribute in `data` containing per-event
                loss weights.
        """
        # Base class constructor
        super().__init__()
        # Check(s)
        if target_labels is None:
            target_labels = self.default_target_labels
        if isinstance(target_labels, str):
            target_labels = [target_labels]

        if prediction_labels is None:
            prediction_labels = self.default_prediction_labels
        if isinstance(prediction_labels, str):
            prediction_labels = [prediction_labels]

        assert isinstance(target_labels, List)  # mypy
        assert isinstance(prediction_labels, List)  # mypy
        # Member variables
        self._regularisation_loss: Optional[float] = None
        self._target_labels = target_labels
        self._prediction_labels = prediction_labels
        self._inference = False
        self._loss_weight = loss_weight

        self._transform_prediction_training: Callable[[Tensor], Tensor] = (
            lambda x: x
        )
        self._transform_prediction_inference: Callable[[Tensor], Tensor] = (
            lambda x: x
        )
        self._transform_target: Callable[[Tensor], Tensor] = lambda x: x
        self._validate_and_set_transforms(
            transform_prediction_and_target,
            transform_target,
            transform_inference,
            transform_support,
        )

    @final
    def _transform_prediction(
        self, prediction: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        if self._inference:
            return self._transform_prediction_inference(prediction)
        else:
            return self._transform_prediction_training(prediction)

    @final
    def inference(self) -> None:
        """Activate inference mode."""
        self._inference = True

    @final
    def train_eval(self) -> None:
        """Deactivate inference mode."""
        self._inference = False

    @final
    def _validate_and_set_transforms(
        self,
        transform_prediction_and_target: Union[Callable, None],
        transform_target: Union[Callable, None],
        transform_inference: Union[Callable, None],
        transform_support: Union[Tuple, None],
    ) -> None:
        """Validate and set transforms.

        Assert that a valid combination of transformation arguments are
        passed and update the corresponding functions.
        """
        # Checks
        assert not (
            (transform_prediction_and_target is not None)
            and (transform_target is not None)
        ), "Please specify at most one of `transform_prediction_and_target` and `transform_target`"  # noqa: E501
        if (transform_target is not None) != (transform_inference is not None):
            self.warning(
                "Setting one of `transform_target` and `transform_inference`, but not "  # noqa: E501
                "the other."
            )

        if transform_target is not None:
            assert transform_target is not None
            assert transform_inference is not None

            if transform_support is not None:
                assert transform_support is not None

                assert (
                    len(transform_support) == 2
                ), "Please specify min and max for transformation support."
                x_test = torch.from_numpy(
                    np.linspace(transform_support[0], transform_support[1], 10)
                )
            else:
                x_test = np.logspace(-6, 6, 12 + 1)
                x_test = torch.from_numpy(
                    np.concatenate([-x_test[::-1], [0], x_test])
                )

            # Add feature dimension before inference transformation to make it
            # match the dimensions of a standard prediction. Remove it again
            # before comparison. Temporary
            try:
                t_test = torch.unsqueeze(transform_target(x_test), -1)
                t_test = torch.squeeze(transform_inference(t_test), -1)
                valid = torch.isfinite(t_test)

                assert torch.allclose(t_test[valid], x_test[valid]), (
                    "The provided transforms for targets during training and "
                    "predictions during inference are not inverse. Please "
                    "adjust transformation functions or support."
                )
                del x_test, t_test, valid

            except IndexError:
                self.warning(
                    "transform_target and/or transform_inference rely on "
                    "indexing, which we won't validate. Please make sure that "
                    "they are mutually inverse, i.e. that\n"
                    "  x = transform_inference(transform_target(x))\n"
                    "for all x that are within your target range."
                )

        # Set transforms
        if transform_prediction_and_target is not None:
            self._transform_prediction_training = (
                transform_prediction_and_target
            )
            self._transform_target = transform_prediction_and_target
        else:
            if transform_target is not None:
                self._transform_target = transform_target
            if transform_inference is not None:
                self._transform_prediction_inference = transform_inference


class LearnedTask(Task):
    """Task class with a learned mapping.

    Applies a learned mapping between the last latent layer of `Model` and
    target space. E.g. the `LearnedTask` contains learnable parameters that
    acts like a prediction head.
    """

    def __init__(
        self,
        hidden_size: int,
        loss_function: "LossFunction",
        **task_kwargs: Any,
    ):
        """Construct `LearnedTask`.

        Args:
            hidden_size: The number of columns in the output of
                         the last latent layer of `Model` using this Task.
                         Available through `Model.nb_outputs`
            loss_function: Loss function appropriate to the task.
        """
        # Base class constructor
        super().__init__(**task_kwargs)

        # Mapping from last hidden layer to required size of input
        self._loss_function = loss_function
        self._affine = Linear(hidden_size, self.nb_inputs)

    @abstractmethod
    def _forward(  # type: ignore
        self, x: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        """Compute loss of `pred` wrt.

        target labels in `data`.
        """

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @final
    def forward(  # type: ignore
        self, x: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        """Forward call for `LearnedTask`.

        The learned embedding transforms last latent layer of Model to
        meet target dimensions.
        """
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        x = self._forward(x=x)
        return self._transform_prediction(x)


class StandardLearnedTask(LearnedTask):
    """Standard class for classification and reconstruction in GraphNeT.

    This class comes with a definition of `compute_loss` that is compatible
    with the vast majority of supervised learning tasks.
    """

    def __init__(
        self,
        hidden_size: int,
        **task_kwargs: Any,
    ):
        """Construct `StandardLearnedTask`.

        Args:
            hidden_size: The number of columns in the output of
                         the last latent layer of `Model` using this Task.
                         Available through `Model.nb_outputs`
        """
        # Base class constructor
        super().__init__(hidden_size=hidden_size, **task_kwargs)

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""

    @final
    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        """Compute supervised learning loss.

        Grabs truth labels in `data` and sends both `pred` and `target` to loss
        function for evaluation. Suits most supervised learning `Task`s.
        """
        target = torch.stack(
            [data[label] for label in self._target_labels], dim=1
        )
        target = self._transform_target(target)
        if self._loss_weight is not None:
            weights = data[self._loss_weight]
        else:
            weights = None
        loss = (
            self._loss_function(pred, target, weights=weights)
            + self._regularisation_loss
        )
        return loss


#######################################################################################
#  EnergyReconstruction

import numpy as np
import torch
from torch import Tensor

from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like



class EnergyReconstruction(StandardLearnedTask):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)

#######################################################################################
#  

#######################################################################################
#  

#######################################################################################
#  

#######################################################################################
#  



