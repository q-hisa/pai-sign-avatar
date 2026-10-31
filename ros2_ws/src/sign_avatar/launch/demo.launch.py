"""手話アバターのデモ: planner / motion_player / avatar_sim / recorder を起動する。

  ros2 launch sign_avatar demo.launch.py texts:="歩いてる人が5人ぐらい"
  ros2 launch sign_avatar demo.launch.py texts:="音声対応は困難です|図で示せますか" run_name:=demo2
録画が終わると（recorder が終了すると）全ノードを止める。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('texts', default_value='歩いてる人が5人ぐらい', description="入力の文。'|'区切りで複数"),
        DeclareLaunchArgument('use_llm', default_value='false', description='LLMで対応づける（要 ANTHROPIC_API_KEY）'),
        DeclareLaunchArgument('lockstep', default_value='true', description='描画完了を待って次のフレームを送る'),
        DeclareLaunchArgument('postprocess', default_value='true', description='録画後に顔を合成してmp4にする'),
        DeclareLaunchArgument('run_name', default_value='', description='出力フォルダ名（空なら日時）'),
    ]
    planner = Node(package='sign_avatar', executable='planner', name='sign_planner', output='screen',
                   parameters=[{'texts': LC('texts'), 'use_llm': LC('use_llm')}])
    player = Node(package='sign_avatar', executable='player', name='sign_motion_player', output='screen',
                  parameters=[{'lockstep': LC('lockstep')}])
    sim = Node(package='sign_avatar', executable='sim', name='avatar_sim', output='screen')
    rec_params = {'postprocess': LC('postprocess'), 'run_name': LC('run_name')}
    recorder = Node(package='sign_avatar', executable='recorder', name='sign_recorder', output='screen',
                    parameters=[rec_params], arguments=[])
    stop_all = RegisterEventHandler(OnProcessExit(target_action=recorder, on_exit=[EmitEvent(event=Shutdown())]))
    return LaunchDescription(args + [recorder, sim, player, planner, stop_all])
