import warnings
from typing import Any, Dict, List, Optional, Union

import torch

from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList
from transformers.generation import validate_stopping_criteria, EosTokenCriteria
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput


class TSGenerationMixin(GenerationMixin):

    def _greedy_search(
            self,
            input_ids: torch.Tensor,
            logits_processor: Optional[LogitsProcessorList] = None,# 用于对 logits 进行后处理（如 temperature, top-k 等）
            stopping_criteria: Optional[StoppingCriteriaList] = None,# 控制生成何时停止（如 max_length, 遇到EOS）
            max_length: Optional[int] = None,# 已弃用，建议使用 stopping_criteria 代替
            pad_token_id: Optional[int] = None,# padding 的 token ID（用于补齐）
            eos_token_id: Optional[Union[int, List[int]]] = None,# 终止的 token ID，遇到即停止生成
            output_attentions: Optional[bool] = None,# 是否输出 attention weights
            output_hidden_states: Optional[bool] = None,# 是否输出隐藏层状态
            output_scores: Optional[bool] = None, # 是否输出每步的 token 概率得分
            output_logits: Optional[bool] = None, # 是否输出原始 logits
            return_dict_in_generate: Optional[bool] = None, # 是否以结构化方式返回（dict 形式）
            synced_gpus: bool = False, # 是否在多 GPU 运行中同步处理
            streamer: Optional["BaseStreamer"] = None, # 用于流式解码，将中间结果实时返回
            **model_kwargs, # 额外的模型输入参数，如 attention_mask、past_key_values 等
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        
        # 记录原始设备并将 input_ids 转到模型所在设备
        input_ids_origin_device = input_ids.device
        input_ids = input_ids.to(self.device)

        # 检查输入维度
        if len(input_ids.shape) == 2:
            batch_size, cur_len = input_ids.shape
        else:
            raise ValueError('Input shape must be: [batch_size, seq_len]')
        
        # 初始化 logits_processor 和 stopping_criteria
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        
         # 处理 max_length（已弃用）转为 stopping_criteria
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        
         # 设置 pad_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        
        # 设置 eos_token_id 和对应的 stopping criteria
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        # 如果是单一 ID，包装为列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # 获取各类输出参数
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

         # 初始化输出容器
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

         # 如果是 encoder-decoder，提取 encoder 的信息
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

         # 追踪未完成的序列
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        max_length = stopping_criteria.max_length

        # 主循环：直到所有序列完成
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            input_length = input_ids.shape[1]

            # #print调度信息
            # print(f"2[Step {input_length}] Remaining: {max_length - input_length} → Using horizon: {max_length - input_length}")

            # 模型前向推理
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                max_horizon_length=max_length - input_length,
            )
            # 多 GPU 情况下同步完成标志
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # 取最后一个 token 的 logits    
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # 处理 logits（如添加温度、过滤等）
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            # 可选保存分数/logits/attn
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
             # 贪婪选择下一 token（注意你此处没用 argmax，而是外部预处理）
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens_scores

            # finished sentences should have their next token be a padding token
            # 已完成序列的输出替换为 pad_token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
             # reshape 成 (B, H, input_size)
            next_tokens = next_tokens.reshape(batch_size, -1, self.config.input_size)
            horizon_length = next_tokens.shape[1]

            # 拼接生成结果
            input_ids = torch.cat([input_ids, next_tokens], dim=-2)

             # 流式输出器
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # 更新缓存等中间变量
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            # 判断当前批中哪些序列已完成
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids[..., 0], scores)
            this_peer_finished = unfinished_sequences.max() == 0

        # 最后截断超长部分
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
        # 停止流式输出器
        if streamer is not None:
            streamer.end()
        # 移回原始设备
        input_ids.squeeze_(dim=-1).to(input_ids_origin_device)

        # 返回结构化输出或 tensor
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids



    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            horizon_length: int = 1,
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
         # 更新 past_key_values 缓存
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

         # 更新其他状态（如 state）
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        # 如果有 token_type_ids，复制最后一个拼接
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            # 非 encoder-decoder，更新 attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], horizon_length))], dim=-1
                )
        else:
            # encoder-decoder 更新 decoder attention mask
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], horizon_length))],
                    dim=-1,
                )
        # 更新 cache_position（如位置编码索引）
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + horizon_length
            # model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs
