# Video Translation Only
# Modified version of app_rvc.py with only Video Translation functionality

import os
import sys
import time
import gradio as gr

from soni_translate.setup import ASR_MODEL_OPTIONS, TRANSLATION_PROCESS_OPTIONS, OUTPUT_TYPE_OPTIONS, COMPUTE_TYPE_GPU, COMPUTE_TYPE_CPU, LANGUAGES_LIST, find_whisper_models
from soni_translate.languages_gui import get_language_data
from soni_translate.utils.diarization import diarization_models
from voice_main import SoniTranslate
from voice_main import create_wav_file_vc

# خواندن متن و متغیرهای زبان
language_data = get_language_data()
lg_conf = language_data["english"]  # Default to English
title = "SoniTranslate - Video Translation"

# ایجاد نمونه SoniTranslate
SoniTr = SoniTranslate()

def create_gui(theme, logs_in_gui=False):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(title)
        gr.Markdown(lg_conf["description"])

        # فقط تب Video translation
        with gr.Tab(lg_conf["tab_translate"]):
            with gr.Row():
                with gr.Column():
                    input_data_type = gr.Dropdown(
                        ["SUBMIT VIDEO", "URL", "Find Video Path"],
                        value="SUBMIT VIDEO",
                        label=lg_conf["video_source"],
                    )

                    def swap_visibility(data_type):
                        if data_type == "URL":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=True, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "SUBMIT VIDEO":
                            return (
                                gr.update(visible=True, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "Find Video Path":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=True, value=""),
                            )

                    video_input = gr.File(
                        label="VIDEO",
                        file_count="multiple",
                        type="filepath",
                    )
                    blink_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["link_label"],
                        info=lg_conf["link_info"],
                        placeholder=lg_conf["link_ph"],
                    )
                    directory_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["dir_label"],
                        info=lg_conf["dir_info"],
                        placeholder=lg_conf["dir_ph"],
                    )
                    input_data_type.change(
                        fn=swap_visibility,
                        inputs=input_data_type,
                        outputs=[video_input, blink_input, directory_input],
                    )

                    gr.HTML()

                    SOURCE_LANGUAGE = gr.Dropdown(
                        LANGUAGES_LIST,
                        value=LANGUAGES_LIST[0],
                        label=lg_conf["sl_label"],
                        info=lg_conf["sl_info"],
                    )
                    TRANSLATE_AUDIO_TO = gr.Dropdown(
                        LANGUAGES_LIST[1:],
                        value="English (en)",
                        label=lg_conf["tat_label"],
                        info=lg_conf["tat_info"],
                    )

                    gr.HTML("<hr></h2>")

                    gr.Markdown(lg_conf["num_speakers"])
                    MAX_TTS = 12
                    min_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=1,
                        label=lg_conf["min_sk"],
                        step=1,
                        visible=False,
                    )
                    max_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=2,
                        step=1,
                        label=lg_conf["max_sk"],
                    )
                    gr.Markdown(lg_conf["tts_select"])

                    def submit(value):
                        visibility_dict = {
                            f"tts_voice{i:02d}": gr.update(visible=i < value)
                            for i in range(MAX_TTS)
                        }
                        return [value for value in visibility_dict.values()]

                    tts_voice00 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk1"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice01 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk2"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice02 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AvaMultilingualNeural-Female",
                        label=lg_conf["sk3"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice03 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-BrianMultilingualNeural-Male",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice04 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-SeraphinaMultilingualNeural-Female",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice05 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-FlorianMultilingualNeural-Male",
                        label=lg_conf["sk6"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice06 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-VivienneMultilingualNeural-Female",
                        label=lg_conf["sk7"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice07 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-RemyMultilingualNeural-Male",
                        label=lg_conf["sk8"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice08 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk9"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice09 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk10"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice10 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk11"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice11 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk12"],
                        visible=False,
                        interactive=True,
                    )
                    max_speakers.change(
                        submit,
                        max_speakers,
                        [
                            tts_voice00,
                            tts_voice01,
                            tts_voice02,
                            tts_voice03,
                            tts_voice04,
                            tts_voice05,
                            tts_voice06,
                            tts_voice07,
                            tts_voice08,
                            tts_voice09,
                            tts_voice10,
                            tts_voice11,
                        ],
                    )

                    with gr.Column():
                        with gr.Accordion(
                            lg_conf["extra_setting"], open=False
                        ):
                            whisper_model_default = (
                                "large-v3"
                                if SoniTr.device == "cuda"
                                else "medium"
                            )

                            WHISPER_MODEL_SIZE = gr.Dropdown(
                                ASR_MODEL_OPTIONS + find_whisper_models(),
                                value=whisper_model_default,
                                label="Whisper ASR model",
                                info=lg_conf["asr_model_info"],
                                allow_custom_value=True,
                            )
                            com_t_opt, com_t_default = (
                                [COMPUTE_TYPE_GPU, "float16"]
                                if SoniTr.device == "cuda"
                                else [COMPUTE_TYPE_CPU, "float32"]
                            )
                            compute_type = gr.Dropdown(
                                com_t_opt,
                                value=com_t_default,
                                label=lg_conf["ctype_label"],
                                info=lg_conf["ctype_info"],
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=8,
                                label=lg_conf["batchz_label"],
                                info=lg_conf["batchz_info"],
                                step=1,
                            )
                            
                            # Setup audio mix options
                            audio_mix_options = [
                                "Mixing audio with sidechain compression",
                                "Adjusting volumes and mixing audio",
                            ]
                            AUDIO_MIX = gr.Dropdown(
                                audio_mix_options,
                                value=audio_mix_options[1],
                                label=lg_conf["aud_mix_label"],
                                info=lg_conf["aud_mix_info"],
                            )

                            # Other settings
                            sub_type_options = [
                                "disable",
                                "srt",
                                "vtt",
                                "ass",
                                "txt",
                                "tsv",
                                "json",
                                "aud",
                            ]
                            sub_type_output = gr.Dropdown(
                                sub_type_options,
                                value=sub_type_options[1],
                                label=lg_conf["sub_type"],
                            )
                            
                            pyannote_models_list = list(
                                diarization_models.keys()
                            )
                            diarization_process_dropdown = gr.Dropdown(
                                pyannote_models_list,
                                value=pyannote_models_list[1],
                                label=lg_conf["diarization_label"],
                            )
                            translate_process_dropdown = gr.Dropdown(
                                TRANSLATION_PROCESS_OPTIONS,
                                value=TRANSLATION_PROCESS_OPTIONS[0],
                                label=lg_conf["tr_process_label"],
                            )
                            
                            # Adding additional required parameters
                            main_output_type = gr.Dropdown(
                                OUTPUT_TYPE_OPTIONS,
                                value=OUTPUT_TYPE_OPTIONS[0],
                                label=lg_conf["out_type_label"],
                            )
                            VIDEO_OUTPUT_NAME = gr.Textbox(
                                label=lg_conf["out_name_label"],
                                value="",
                                info=lg_conf["out_name_info"],
                            )

                with gr.Column(variant="compact"):
                    edit_sub_check = gr.Checkbox(
                        label=lg_conf["edit_sub_label"],
                        info=lg_conf["edit_sub_info"],
                    )
                    dummy_false_check = gr.Checkbox(
                        False,
                        visible=False,
                    )

                    def visible_component_subs(input_bool):
                        if input_bool:
                            return gr.update(visible=True), gr.update(
                                visible=True
                            )
                        else:
                            return gr.update(visible=False), gr.update(
                                visible=False
                            )

                    subs_button = gr.Button(
                        lg_conf["button_subs"],
                        variant="primary",
                        visible=False,
                    )
                    subs_edit_space = gr.Textbox(
                        visible=False,
                        lines=10,
                        label=lg_conf["editor_sub_label"],
                        info=lg_conf["editor_sub_info"],
                        placeholder=lg_conf["editor_sub_ph"],
                    )
                    edit_sub_check.change(
                        visible_component_subs,
                        [edit_sub_check],
                        [subs_button, subs_edit_space],
                    )

                    with gr.Row():
                        video_button = gr.Button(
                            lg_conf["button_translate"],
                            variant="primary",
                        )
                    with gr.Row():
                        video_output = gr.File(
                            label=lg_conf["output_result_label"],
                            file_count="multiple",
                            interactive=False,
                        )

                    gr.HTML("<hr></h2>")

                    if (
                        os.getenv("YOUR_HF_TOKEN") is None
                        or os.getenv("YOUR_HF_TOKEN") == ""
                    ):
                        HFKEY = gr.Textbox(
                            visible=True,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )
                    else:
                        HFKEY = gr.Textbox(
                            visible=False,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )

        # معرفی متغیرهای مورد نیاز دیگر
        volume_original_mix = gr.Slider(value=0.25, visible=False)
        volume_translated_mix = gr.Slider(value=1.80, visible=False)
        audio_accelerate = gr.Slider(value=1.9, visible=False)
        acceleration_rate_regulation_gui = gr.Checkbox(False, visible=False)
        avoid_overlap_gui = gr.Checkbox(False, visible=False)
        vocal_refinement_gui = gr.Checkbox(False, visible=False)
        literalize_numbers_gui = gr.Checkbox(True, visible=False)
        segment_duration_limit_gui = gr.Slider(value=15, visible=False)
        input_srt = gr.File(visible=False)
        main_voiceless_track = gr.Checkbox(False, visible=False)
        voice_imitation_gui = gr.Checkbox(False, visible=False)
        voice_imitation_max_segments_gui = gr.Slider(value=3, visible=False)
        voice_imitation_vocals_dereverb_gui = gr.Checkbox(False, visible=False)
        voice_imitation_remove_previous_gui = gr.Checkbox(True, visible=False)
        voice_imitation_method_gui = gr.Dropdown(value="openvoice", visible=False)
        wav_speaker_dereverb = gr.Checkbox(True, visible=False)
        text_segmentation_scale_gui = gr.Dropdown(value="sentence", visible=False)
        divide_text_segments_by_gui = gr.Textbox(value="", visible=False)
        soft_subtitles_to_video_gui = gr.Checkbox(False, visible=False)
        burn_subtitles_to_video_gui = gr.Checkbox(False, visible=False)
        enable_cache_gui = gr.Checkbox(True, visible=False)
        enable_custom_voice = gr.Checkbox(False, visible=False)
        workers_custom_voice = gr.Slider(value=1, visible=False)
        is_gui_dummy_check = gr.Checkbox(True, visible=False)
        PREVIEW = gr.Checkbox(False, visible=False)

        # Run translate text
        subs_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                video_input,
                blink_input,
                directory_input,
                HFKEY,
                PREVIEW,
                WHISPER_MODEL_SIZE,
                batch_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                edit_sub_check,  # TRUE BY DEFAULT
                dummy_false_check,  # dummy false
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=subs_edit_space,
        )

        # Run translate tts and complete
        video_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                video_input,
                blink_input,
                directory_input,
                HFKEY,
                PREVIEW,
                WHISPER_MODEL_SIZE,
                batch_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                dummy_false_check,  # get_translated_text
                dummy_false_check,  # get_video_from_text
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=[video_output],
            cache_examples=False,
        )

    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable public link",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Displays the operations performed in Logs",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="Select the language of the interface",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Soft",
        help="Select the theme of the interface",
    )
    
    args = parser.parse_args()
    
    # Set language
    lg_conf = language_data.get(args.language, language_data["english"])
    
    demo = create_gui(args.theme, args.logs)
    demo.queue(max_size=15)
    demo.launch(share=args.share, enable_queue=True) 