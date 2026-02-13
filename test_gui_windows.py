# test_gui_windows.py
"""
测试Windows下MetaWorld仿真界面能否打开
"""
import sys
import os
import cv2
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.policies import SawyerReachV3Policy

def test_gui():
    """测试GUI界面"""
    print("=" * 50)
    print("测试Windows下MetaWorld仿真界面")
    print("=" * 50)
    
    # 创建环境
    env_name = "reach-v3"
    print(f"\n1. 创建环境: {env_name}")
    
    try:
        env = ALL_V3_ENVIRONMENTS[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(0)
        print("✓ 环境创建成功")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return
    
    # 创建策略
    print(f"\n2. 创建专家策略")
    try:
        policy = SawyerReachV3Policy()
        print("✓ 策略创建成功")
    except Exception as e:
        print(f"✗ 策略创建失败: {e}")
        return
    
    # 运行测试
    print(f"\n3. 开始运行仿真（按ESC键退出）")
    print("   如果看到窗口弹出，说明GUI正常工作")
    print("=" * 50)
    
    try:
        # 重置环境
        env.reset()
        env.reset_model()
        env.render_mode = 'rgb_array'
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            o = reset_result[0]
        else:
            o = reset_result
        
        episode_num = 0
        max_episodes = 3  # 测试3个episode
        
        while episode_num < max_episodes:
            print(f"\nEpisode {episode_num + 1}/{max_episodes}")
            
            step = 0
            max_steps = 500
            
            while step < max_steps:
                # 获取动作
                a = policy.get_action(o)
                a = np.clip(a, env.action_space.low, env.action_space.high)
                
                # 执行动作
                step_result = env.step(a)
                if len(step_result) == 5:
                    o, r, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    o, r, done, info = step_result
                
                # 渲染并显示
                img = env.render()[:, :, ::-1]
                img = cv2.resize(img, (224, 224)).astype(np.uint8)
                
                cv2.imshow("MetaWorld Simulation", img)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    print("用户按下ESC键，退出测试")
                    return
                
                # 检查是否成功
                if info.get("success", False):
                    print(f"  ✓ 任务成功！奖励: {r:.2f}")
                    break
                
                step += 1
            
            episode_num += 1
            
            # 重置环境
            env.reset()
            env.reset_model()
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                o = reset_result[0]
            else:
                o = reset_result
        
        print("\n" + "=" * 50)
        print("测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n已关闭所有窗口")

if __name__ == "__main__":
    test_gui()