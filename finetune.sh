
###
 # @Author: riverman nanjing.com
 # @Date: 2025-04-04 00:47:32
 # @LastEditors: riverman nanjing.com
 # @LastEditTime: 2025-04-04 00:47:33
 # @FilePath: /wsj/bliptime/blip2_finetune_demo/finetune.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 




python main_parallelism.py --epochs 3 --batch_size 16 > finetune_log.txt 2>&1