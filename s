[33mcommit 8945db84585246de21ca09ea62b58b51f0ace774[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: loveaoe33 <loveaoe33@gmail.com>
Date:   Wed Sep 6 16:41:36 2023 +0800

    20230906

[33mcommit 2334aa733872bc4bb3e1a1ba90e5fd319399596f[m[33m ([m[1;31morigins/master[m[33m)[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Sun Jun 18 16:09:41 2023 +0200

    Uninstall `wandb` from notebook environments (#11730)
    
    * Uninstall `wandb` from notebook environments
    
    Due to undesired behavior in https://www.kaggle.com/code/ultralytics/yolov8/comments#2306977
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    * fix import
    
    * [pre-commit.ci] auto fixes from pre-commit.com hooks
    
    for more information, see https://pre-commit.ci
    
    ---------
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

[33mcommit f616dc5af217a7193d80b79e119d7a40798750ad[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Sun Jun 18 15:44:56 2023 +0200

    Uninstall `wandb` from notebook environments (#11729)
    
    Uninstall W&B that are present in notebooks
    
    Resolves unwanted W&B install issues in https://www.kaggle.com/code/ultralytics/yolov8/comments#2306977
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 878d9c8d5b21253ee3a086b69a94fbbf55e56088[m
Author: hackerrajeshkumar <120269593+hackerrajeshkumar@users.noreply.github.com>
Date:   Sun Jun 18 00:21:50 2023 +0530

    Update export.py (#11638)
    
    * Update export.py
    
    Signed-off-by: hackerrajeshkumar <120269593+hackerrajeshkumar@users.noreply.github.com>
    
    * Update export.py
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    ---------
    
    Signed-off-by: hackerrajeshkumar <120269593+hackerrajeshkumar@users.noreply.github.com>
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    Co-authored-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 7e2139256143e2ae8befc392379f63cf97c3c061[m
Author: 琪亚娜芽衣贴贴 <39751846+kisaragychihaya@users.noreply.github.com>
Date:   Sun Jun 18 02:50:10 2023 +0800

    Add OpenVINO NNCF Support (Using --int8 flag) (#11706)
    
    * Add OpenVINO NNCF support
    
    * Add openvino to flag help text
    
    Using --int8  --data  your_dataset.yaml to quant your ov model
    
    Signed-off-by: 琪亚娜芽衣贴贴 <39751846+kisaragychihaya@users.noreply.github.com>
    
    * [pre-commit.ci] auto fixes from pre-commit.com hooks
    
    for more information, see https://pre-commit.ci
    
    * Update export.py
    
    Redundant
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    ---------
    
    Signed-off-by: 琪亚娜芽衣贴贴 <39751846+kisaragychihaya@users.noreply.github.com>
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>
    Co-authored-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 3812a1a29f7874a370967eca1bd77a69820df88f[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Thu Jun 15 21:15:59 2023 +0200

    Update Discord invite URLs (#11713)

[33mcommit 9bb50b4ffee5fcdcfd381ac2b885d1303c767650[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Thu Jun 15 13:52:17 2023 +0200

    Remove Python 3.7 from tests (#11708)
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 98acd111b110a60843291edcf95e708d73abfe5d[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Thu Jun 15 13:49:19 2023 +0200

    Update Comet integration (#11648)
    
    * Update Comet
    
    * Update Comet
    
    * Update Comet
    
    * Add default Experiment Name
    
    * [pre-commit.ci] auto fixes from pre-commit.com hooks
    
    for more information, see https://pre-commit.ci
    
    * Update tutorial.ipynb
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    * Update tutorial.ipynb
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    * Update tutorial.ipynb
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    * Update tutorial.ipynb
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    ---------
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

[33mcommit a199480ba6bb527598df11abbc1d679ccda82670[m
Author: wuhongsheng <664116298@qq.com>
Date:   Thu Jun 8 05:28:01 2023 +0800

    Fix the bug that tensorRT batch_size does not take effect (#11672)
    
    * Fix the bug that tensorRT batch_size does not take effect
    
    Signed-off-by: wuhongsheng <664116298@qq.com>
    
    * [pre-commit.ci] auto fixes from pre-commit.com hooks
    
    for more information, see https://pre-commit.ci
    
    ---------
    
    Signed-off-by: wuhongsheng <664116298@qq.com>
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

[33mcommit 89c3040e734e8a0185fb49c667184600bb827f25[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Tue Jun 6 14:48:13 2023 +0200

    Fix OpenVINO export (#11666)
    
    * Fix OpenVINO export
    
    Resolves https://github.com/ultralytics/yolov5/issues/11645
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    * [pre-commit.ci] auto fixes from pre-commit.com hooks
    
    for more information, see https://pre-commit.ci
    
    * Update export.py
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    
    ---------
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

[33mcommit 76ea9ed3a4d42fe835e172672132f13cf5286648[m
Author: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>
Date:   Tue Jun 6 00:29:58 2023 +0200

    [pre-commit.ci] pre-commit suggestions (#11661)
    
    updates:
    - [github.com/asottile/pyupgrade: v3.3.2 → v3.4.0](https://github.com/asottile/pyupgrade/compare/v3.3.2...v3.4.0)
    
    Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

[33mcommit 5f11555e0796f9471185a16dc79860c236f656f8[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Thu Jun 1 13:06:22 2023 +0200

    Update ci-testing.yml (#11642)
    
    Signed-off-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 5eb7f7ddc034b9ad07578a6f954c58170579ebc6[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Wed May 31 12:22:52 2023 +0200

    Update requirements.txt `ultralytics>=8.0.111` (#11630)

[33mcommit 573334200866d5400325c0e8430c7154f7f23a59[m
Author: Snyk bot <snyk-bot@snyk.io>
Date:   Tue May 30 17:32:46 2023 +0100

    [Snyk] Security upgrade numpy from 1.21.3 to 1.22.2 (#11531)
    
    fix: requirements.txt to reduce vulnerabilities
    
    
    The following vulnerabilities are fixed by pinning transitive dependencies:
    - https://snyk.io/vuln/SNYK-PYTHON-NUMPY-2321964
    - https://snyk.io/vuln/SNYK-PYTHON-NUMPY-2321966
    - https://snyk.io/vuln/SNYK-PYTHON-NUMPY-2321970

[33mcommit c3c130416323f3766d4abe95c2ff88bc9e2264dd[m
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Tue May 23 12:12:30 2023 +0200

    Update LinkedIn URL (#11576)

[33mcommit 6e04b94fa9fb12ff66b2329660de8a5a8e5f1b1d[m
Author: Peter van Lunteren <contact@pvanlunteren.com>
Date:   Mon May 22 14:12:10 2023 +0200

    add smoothing line to results.png to improve readability (#11536)
    
    Signed-off-by: Peter van Lunteren <contact@pvanlunteren.com>
    Co-authored-by: Glenn Jocher <glenn.jocher@ultralytics.com>

[33mcommit 4298c5dc3aa5c9a20f4e95e3a350903994e2e75e[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Mon May 22 14:09:35 2023 +0200

    Bump slackapi/slack-github-action from 1.23.0 to 1.24.