from loguru import logger
from typing import List
from pathlib import Path


class HarborTaskDataset:
    """
    A dataset that loads Harbor task data from direct file/directory paths.
    Each dataset item is a path to a task directory.
    """

    def __init__(
        self,
        data_files: List[str],
    ):
        """
        Initialize the HarborTaskDataset.

        Args:
            data_files: List of direct file/directory paths pointing to Harbor task data
        """
        self.data_files = data_files

        # Load all data files
        self.task_paths = self._load_data_files()

        logger.info(f"HarborTaskDataset initialized with {len(self.task_paths)} task paths")

    def _load_data_files(self) -> List[Path]:
        """Load all data files from direct paths and return list of task paths."""
        task_paths = []

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue

            logger.info(f"Loading data from: {data_source}")

            # If the path is a directory, find all valid task subdirectories
            if source_path.is_dir():
                # Look for task subdirectories and validate them
                all_dirs = [d for d in source_path.iterdir() if d.is_dir()]
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    task_paths.extend(valid_task_dirs)
                    logger.info(
                        f"Found {len(valid_task_dirs)} valid task directories out of {len(all_dirs)} total directories"
                    )
                elif self._is_valid_task_directory(source_path):
                    # If no subdirectories but the main directory is valid, treat it as a task
                    task_paths.append(source_path)
                    logger.info("Using main directory as valid task")
                else:
                    logger.warning(f"No valid task directories found in {source_path}")
            else:
                # If it's a file, treat it as a single task (files can't be valid task directories)
                logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")

        return task_paths

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False

        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return {
            "prompt": str(self.task_paths[index]),
            "env_class": None,
            "env_extras": {"data_source": str(self.task_paths[index])},
            "uid": str(index),
        }

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)

    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for index, task_path in enumerate(self.task_paths):
            yield {
                "prompt": str(task_path),
                "env_class": None,
                "env_extras": {"data_source": str(task_path)},
                "uid": str(index),
            }

    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list


class TTTHarborOneTaskDataset(HarborTaskDataset):
    """Expose one physical Harbor task as a virtual dataset.

    SkyRL expects a training batch to contain train_batch_size distinct prompt
    UIDs. Reusing UID "0" would make downstream group-aware minibatching treat
    all virtual prompts as one prompt, so each virtual row gets its own stable
    UID while still pointing at the same task directory.
    """

    def __init__(self, data_files: List[str], virtual_size: int):
        super().__init__(data_files=data_files)
        if len(self.task_paths) != 1:
            raise ValueError(
                "TTTHarborOneTaskDataset requires exactly one valid Harbor task, "
                f"but found {len(self.task_paths)}"
            )
        if virtual_size < 1:
            raise ValueError(f"virtual_size must be positive, got {virtual_size}")
        self.virtual_size = int(virtual_size)
        logger.info(
            "Repeating Harbor task {} over {} virtual dataset rows",
            self.task_paths[0],
            self.virtual_size,
        )

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self.virtual_size
        if index < 0 or index >= self.virtual_size:
            raise IndexError(
                f"Index {index} out of range for virtual dataset of size {self.virtual_size}"
            )
        task_path = str(self.task_paths[0])
        return {
            "prompt": task_path,
            "env_class": None,
            "env_extras": {"data_source": task_path},
            "uid": f"ttt-{index}",
        }

    def __iter__(self):
        for index in range(self.virtual_size):
            yield self[index]
