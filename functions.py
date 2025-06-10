import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from audiorecorder import audiorecorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct

def ensure_directory_exists(directory_path):
    """
    ディレクトリが存在しない場合は作成する
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def safe_remove_file(file_path):
    """
    ファイルが存在する場合のみ削除する
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.warning(f"ファイル削除に失敗しました: {file_path}")
        print(f"File deletion error: {e}")

def record_audio(audio_input_file_path):
    """
    音声入力を受け取って音声ファイルを作成
    """
    # 入力ディレクトリの存在確認
    ensure_directory_exists(ct.AUDIO_INPUT_DIR)
    
    audio = audiorecorder(
        start_prompt="発話開始",
        pause_prompt="やり直す",
        stop_prompt="発話終了",
        start_style={"color":"white", "background-color":"black"},
        pause_style={"color":"gray", "background-color":"white"},
        stop_style={"color":"white", "background-color":"black"}
    )

    if len(audio) > 0:
        try:
            audio.export(audio_input_file_path, format="wav")
        except Exception as e:
            st.error(f"音声ファイルの保存に失敗しました: {e}")
            st.stop()
    else:
        st.stop()

def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """
    try:
        with open(audio_input_file_path, 'rb') as audio_input_file:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=audio_input_file,
                language="en"
            )
        
        # 音声入力ファイルを削除
        safe_remove_file(audio_input_file_path)
        
        return transcript
    except Exception as e:
        st.error(f"音声の文字起こしに失敗しました: {e}")
        safe_remove_file(audio_input_file_path)
        st.stop()

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """
    # 出力ディレクトリの存在確認
    ensure_directory_exists(ct.AUDIO_OUTPUT_DIR)
    
    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    
    try:
        with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
            temp_audio_output_file.write(llm_response_audio)
        
        audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
        audio_mp3.export(audio_output_file_path, format="wav")

        # 音声出力用に一時的に作ったmp3ファイルを削除
        safe_remove_file(temp_audio_output_filename)
        
    except Exception as e:
        st.error(f"音声ファイルの変換に失敗しました: {e}")
        safe_remove_file(temp_audio_output_filename)
        raise e

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """
    try:
        # 音声ファイルの存在確認
        if not os.path.exists(audio_output_file_path):
            st.error(f"音声ファイルが見つかりません: {audio_output_file_path}")
            return

        # 音声ファイルの読み込み
        audio = AudioSegment.from_wav(audio_output_file_path)
        
        # 速度を変更
        if speed != 1.0:
            # frame_rateを変更することで速度を調整
            modified_audio = audio._spawn(
                audio.raw_data, 
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

            modified_audio.export(audio_output_file_path, format="wav")

        # PyAudioで再生
        with wave.open(audio_output_file_path, 'rb') as play_target_file:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(play_target_file.getsampwidth()),
                channels=play_target_file.getnchannels(),
                rate=play_target_file.getframerate(),
                output=True
            )

            data = play_target_file.readframes(1024)
            while data:
                stream.write(data)
                data = play_target_file.readframes(1024)

            stream.stop_stream()
            stream.close()
            p.terminate()
        
    except Exception as e:
        st.error(f"音声再生に失敗しました: {e}")
    finally:
        # LLMからの回答の音声ファイルを削除
        safe_remove_file(audio_output_file_path)

def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chain = ConversationChain(
            llm=st.session_state.llm,
            memory=st.session_state.memory,
            prompt=prompt
        )

        return chain
    except Exception as e:
        st.error(f"Chain作成に失敗しました: {e}")
        st.stop()

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    """
    try:
        # 問題文を生成するChainを実行し、問題文を取得
        problem = st.session_state.chain_create_problem.predict(input="")

        # LLMからの回答を音声データに変換
        llm_response_audio = st.session_state.openai_obj.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=problem
        )

        # 音声ファイルの作成
        audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
        save_to_wav(llm_response_audio.content, audio_output_file_path)

        # 音声ファイルの読み上げ
        play_wav(audio_output_file_path, st.session_state.speed)

        return problem, llm_response_audio
        
    except Exception as e:
        st.error(f"問題生成・音声再生に失敗しました: {e}")
        st.stop()

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """
    try:
        llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")
        return llm_response_evaluation
    except Exception as e:
        st.error(f"評価生成に失敗しました: {e}")
        st.stop()
        