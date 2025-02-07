# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from fillerbuster.configs.ablations import AblationMethods
from fillerbuster.configs.base import BaseMethods

Methods = Union[BaseMethods, AblationMethods]
