import yaml


class DataConfig:
    def __init__(self, batch_size: int = 32, num_workers: int = 4, min_str_len: int = 1, max_str_len: int = 100, train_samples: int = 10000, val_samples: int = 2000, test_samples: int = 2000):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_str_len = min_str_len
        self.max_str_len = max_str_len
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

class ModelConfig:
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, max_seq_len: int = 100):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len


class OptimConfig:
    def __init__(self, lr: float = 1e-4):
        self.lr = lr

class TrainerConfig:
    def __init__(self, epochs: int = 10, patience: int = 3):
        self.epochs = epochs
        self.patience = patience


class EvalConfig:
    def __init__(self, sample_strings: list[str] | None = None, ood_lengths: list[int] | None = None):
        self.sample_strings = sample_strings or ["abcdef", "hello", "xyz", "python", "ignite", "gsoc"]
        self.ood_lengths = ood_lengths or [12, 15, 18, 20]


class OutputConfig:
    def __init__(self, root_dir: str = "runs", save_checkpoints: bool = True, n_saved: int = 1):
        self.root_dir = root_dir
        self.save_checkpoints = save_checkpoints
        self.n_saved = n_saved

class ExperimentConfig:
    def __init__(
        self,
        run_name: str = "default_run",
        seed: int = 42,
        device: str = "auto",
        data_config: DataConfig | None = None,
        model_config: ModelConfig | None = None,
        optim_config: OptimConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        eval_config: EvalConfig | None = None,
        output_config: OutputConfig | None = None,
    ):
        self.run_name = run_name
        self.seed = seed
        self.device = device
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.optim_config = optim_config or OptimConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.eval_config = eval_config or EvalConfig()
        self.output_config = output_config or OutputConfig()


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a YAML object at the top level.")

    data_config = DataConfig(**config_dict["data"])
    model_config = ModelConfig(**config_dict["model"])
    optim_config = OptimConfig(**config_dict["optim"])
    trainer_config = TrainerConfig(**config_dict["trainer"])
    eval_config = EvalConfig(**config_dict["eval"])
    output_config = OutputConfig(**config_dict["output"])
    
    built_config = ExperimentConfig(
        run_name=config_dict["run_name"],
        seed=config_dict["seed"],
        device=config_dict["device"],
        data_config=data_config,
        model_config=model_config,
        optim_config=optim_config,
        trainer_config=trainer_config,
        eval_config=eval_config,
        output_config=output_config
    )

    validate_config(built_config)

    return built_config


def validate_config(config: ExperimentConfig) -> None:
    if config.data_config.batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    if config.data_config.num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")
    if config.data_config.min_str_len <= 0 or config.data_config.max_str_len <= 0:
        raise ValueError("String lengths must be positive.")
    if config.data_config.min_str_len > config.data_config.max_str_len:
        raise ValueError("min_str_len must be <= max_str_len.")
    if config.data_config.train_samples <= 0:
        raise ValueError("train_samples must be positive.")
    if config.data_config.val_samples <= 0:
        raise ValueError("val_samples must be positive.")
    if config.data_config.test_samples <= 0:
        raise ValueError("test_samples must be positive.")
    if config.model_config.d_model <= 0:
        raise ValueError("Model dimension must be positive.")
    if config.model_config.nhead <= 0:
        raise ValueError("Number of heads must be positive.")
    if config.model_config.d_model % config.model_config.nhead != 0:
        raise ValueError("d_model must be divisible by nhead.")
    if config.model_config.num_layers <= 0:
        raise ValueError("Number of layers must be positive.")
    if config.model_config.dim_feedforward <= 0:
        raise ValueError("Feedforward dimension must be positive.")
    if not (0 <= config.model_config.dropout < 1):
        raise ValueError("Dropout must be in the range [0, 1).")
    if config.model_config.max_seq_len <= 0:
        raise ValueError("Max sequence length must be positive.")
    if config.optim_config.lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if config.trainer_config.epochs <= 0:
        raise ValueError("Number of epochs must be positive.")
    if config.trainer_config.patience < 0:
        raise ValueError("Patience cannot be negative.")
    if len(config.eval_config.sample_strings) == 0:
        raise ValueError("sample_strings must be non-empty.")
    if any(length <= 0 for length in config.eval_config.ood_lengths):
        raise ValueError("All ood_lengths must be positive.")
    if config.output_config.n_saved <= 0:
        raise ValueError("output.n_saved must be positive.")