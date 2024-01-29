#!/usr/bin/env bash

# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ $OAP_MLLIB_TESTING == "true" ]]; then
  exit 0
fi

if [[ -z $ONEDAL_VERSION ]]; then
  if [[ -e $DAALROOT ]]; then
    export ONEDAL_VERSION=$(echo "$DAALROOT" | awk -F '/' '{print $(NF)}')
  elif [[ -e $DALROOT ]]; then
    export ONEDAL_VERSION=$(echo "$DALROOT" | awk -F '/' '{print $(NF)}')
  else
    echo DAALROOT not defined!
    exit 1
  fi
fi

verlte() {
    printf '%s\n%s' "$1" "$2" | sort -C -V
}
# Reference version
reference_version="2023.2.0"

# Compare versions
verlte "$ONEDAL_VERSION" "$reference_version" && echo "1" || echo "0"
result=$?

#Check the result of the comparison
if [ "$result" -eq 0 ]; then
    echo "$ONEDAL_VERSION is greater than $reference_version"
    make clean
    make -f Makefile -j
elif [ "$result" -eq 1 ]; then
    echo "$ONEDAL_VERSION is less than or equal $reference_version"
    make clean
    make -f Makefile_2023.2.0 -j
fi
