#!/bin/bash
# Launch the scene-graph student training (scripts/remote_train_scene_lora.py) on Colab.
#
#   scripts/colab_train_scene.sh SESSION SMOKE ARCHIVES_DIR
#
# SESSION is an existing Colab session (colab --auth=adc new -s SESSION --gpu L4), SMOKE is
# 1 for the 4-step check or 0 for the real run, ARCHIVES_DIR holds code.tar, train.tar.gz,
# teacher.tar.gz, relations.tar.gz and subjects.tar.gz (see the training script's header).
# The run continues on the Colab machine after this script returns; read /content/train.log.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
cd "$3" || exit 1
c() { colab --auth=adc "$@"; }
for f in code.tar train.tar.gz teacher.tar.gz relations.tar.gz subjects.tar.gz; do
  c upload -s "$1" "$f" "/content/$f" || exit 1
done
c upload -s "$1" "$here/remote_train_scene_lora.py" /content/train.py || exit 1
echo "import subprocess; subprocess.Popen('rm -rf /content/cay /content/student; CAY_SMOKE=$2 nohup python /content/train.py > /content/train.log 2>&1 &', shell=True); print('started')" \
  | c exec -s "$1"
