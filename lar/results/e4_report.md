# E4：文本域 × 指标形式

## 1. 覆盖审计

主结论仅使用 MLLM_Avg（实际 n=79）；matched 与 PC1 仅作辅助且池完全相同。

| 目标 | 文本域 | 指标 | n | family 构成 |
|---|---|---|---:|---|
| MLLM_Avg | caption | VSA | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | caption | mutual_kNN_k5 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | caption | mutual_kNN_k10 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | caption | mutual_kNN_k20 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | caption | cm_cka | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | caption | cm_r2 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg_matched | caption | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | caption | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | caption | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | caption | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | caption | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | caption | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | caption | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg | answer_other | VSA | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_other | mutual_kNN_k5 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_other | mutual_kNN_k10 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_other | mutual_kNN_k20 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_other | cm_cka | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_other | cm_r2 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg_matched | answer_other | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_other | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_other | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_other | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_other | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_other | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_other | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg | question_other | VSA | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | question_other | mutual_kNN_k5 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | question_other | mutual_kNN_k10 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | question_other | mutual_kNN_k20 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | question_other | cm_cka | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | question_other | cm_r2 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg_matched | question_other | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | question_other | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | question_other | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | question_other | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | question_other | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | question_other | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | question_other | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg | qa_concat | VSA | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | qa_concat | mutual_kNN_k5 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | qa_concat | mutual_kNN_k10 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | qa_concat | mutual_kNN_k20 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | qa_concat | cm_cka | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | qa_concat | cm_r2 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg_matched | qa_concat | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | qa_concat | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | qa_concat | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | qa_concat | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | qa_concat | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | qa_concat | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | qa_concat | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg | answer_all_types | VSA | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_all_types | mutual_kNN_k5 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_all_types | mutual_kNN_k10 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_all_types | mutual_kNN_k20 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_all_types | cm_cka | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg | answer_all_types | cm_r2 | 79 | clip:1, dino:4, dinov2:4, dinov3:1, eupe:6, ijepa:1, mc1:9, mc2:15, pe_core:5, pe_lang:3, pixio:4, raev2:1, siglip2:15, toklip:2, uniar:1, unitok:1, vilau:1, webssl_mae:5 |
| MLLM_Avg_matched | answer_all_types | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_all_types | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_all_types | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_all_types | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_all_types | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg_matched | answer_all_types | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | VSA | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | mutual_kNN_k5 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | mutual_kNN_k10 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | mutual_kNN_k20 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | cm_cka | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| PC1 | answer_all_types | cm_r2 | 22 | clip:1, ijepa:1, mc1:4, mc2:4, raev2:1, siglip2:11 |
| MLLM_Avg | eval_answer | VSA | 0 | NA |
| MLLM_Avg | eval_answer | mutual_kNN_k5 | 0 | NA |
| MLLM_Avg | eval_answer | mutual_kNN_k10 | 0 | NA |
| MLLM_Avg | eval_answer | mutual_kNN_k20 | 0 | NA |
| MLLM_Avg | eval_answer | cm_cka | 0 | NA |
| MLLM_Avg | eval_answer | cm_r2 | 0 | NA |
| MLLM_Avg_matched | eval_answer | VSA | 0 | NA |
| MLLM_Avg_matched | eval_answer | mutual_kNN_k5 | 0 | NA |
| MLLM_Avg_matched | eval_answer | mutual_kNN_k10 | 0 | NA |
| MLLM_Avg_matched | eval_answer | mutual_kNN_k20 | 0 | NA |
| MLLM_Avg_matched | eval_answer | cm_cka | 0 | NA |
| MLLM_Avg_matched | eval_answer | cm_r2 | 0 | NA |
| PC1 | eval_answer | VSA | 0 | NA |
| PC1 | eval_answer | mutual_kNN_k5 | 0 | NA |
| PC1 | eval_answer | mutual_kNN_k10 | 0 | NA |
| PC1 | eval_answer | mutual_kNN_k20 | 0 | NA |
| PC1 | eval_answer | cm_cka | 0 | NA |
| PC1 | eval_answer | cm_r2 | 0 | NA |

## 2. 主表（MLLM_Avg）

格内为：全表 rho / 一族一个 rho / top-1 regret。

| 文本域 | VSA | mutual-kNN k=10 | linear CKA | ridge R² |
|---|---:|---:|---:|---:|
| caption | 0.117/0.071/8.565 | 0.590/0.427/2.403 | 0.229/0.261/7.314 | -0.564/-0.103/13.042 |
| answer_other | 0.592/0.562/4.487 | 0.665/0.416/2.167 | 0.433/0.116/4.405 | -0.624/-0.169/13.686 |
| question_other | -0.207/-0.184/10.966 | 0.572/0.478/2.903 | 0.370/0.356/4.802 | -0.626/-0.162/13.626 |
| qa_concat | -0.136/-0.169/9.899 | 0.610/0.441/1.973 | 0.330/0.351/5.624 | -0.628/-0.163/13.685 |
| answer_all_types | 0.372/0.450/5.870 | 0.622/0.382/2.662 | 0.494/0.123/4.097 | -0.647/-0.193/13.636 |
| eval_answer | NA/NA/NA | NA/NA/NA | NA/NA/NA | NA/NA/NA |

mutual-kNN 的 k=5/20 完整数值及三个目标均保存在 `e4.json`。

## 3. 换文本还是换指标

var_row=0.03149，var_col=0.30504，比值=0.10：**不支持“换文本比换指标重要”**。
双因素方差份额：文本域=0.037，指标=0.894，交互=0.069。这是无重复单元格分解，不报告 F 检验。

## 4. 机制诊断

| 文本域 | N | eff_rank | RankMe | 平均余弦 | top1 能量 | token 长度 | 唯一文本 | CCA1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | 4618 | 26.942 | 870.160 | 0.937 | 0.122 | 11.805 | 4590 | 0.925 |
| answer_other | 4618 | 15.467 | 605.668 | 0.783 | 0.207 | 1.671 | 1687 | 0.649 |
| question_other | 4618 | 19.803 | 789.076 | 0.964 | 0.155 | 7.597 | 3714 | 0.776 |
| qa_concat | 4618 | 20.785 | 820.250 | 0.963 | 0.142 | 8.978 | 4302 | 0.814 |
| answer_all_types | 4618 | 7.053 | 431.461 | 0.740 | 0.315 | 1.409 | 1193 | 0.486 |
| eval_answer | NA | NA | NA | NA | NA | NA | NA | NA |

CCA 前 20 维的均值、标准差和模型数见 `e4.json`。
预设机制判据成立：False。

caption all-but-the-top 消融（全表 rho）：

- VSA_abtt1: 0.160
- VSA_abtt2: 0.199
- VSA_abtt3: 0.213

## 5. 组合 regret

| 文本域 | 指标 | probe | probe+指标 | 改善 |
|---|---|---:|---:|---:|
| caption | VSA | 1.622 | 1.562 | 0.061 |
| caption | mutual_kNN_k5 | 1.690 | 1.809 | -0.119 |
| caption | mutual_kNN_k10 | 1.676 | 1.798 | -0.122 |
| caption | mutual_kNN_k20 | 1.652 | 1.810 | -0.159 |
| caption | cm_cka | 1.725 | 1.786 | -0.061 |
| caption | cm_r2 | 1.732 | 2.140 | -0.408 |
| answer_other | VSA | 1.616 | 1.328 | 0.288 |
| answer_other | mutual_kNN_k5 | 1.675 | 2.019 | -0.344 |
| answer_other | mutual_kNN_k10 | 1.688 | 1.973 | -0.285 |
| answer_other | mutual_kNN_k20 | 1.793 | 1.969 | -0.176 |
| answer_other | cm_cka | 1.704 | 1.714 | -0.009 |
| answer_other | cm_r2 | 1.624 | 2.081 | -0.457 |
| question_other | VSA | 1.558 | 1.639 | -0.081 |
| question_other | mutual_kNN_k5 | 1.685 | 1.657 | 0.028 |
| question_other | mutual_kNN_k10 | 1.683 | 1.711 | -0.028 |
| question_other | mutual_kNN_k20 | 1.677 | 1.713 | -0.035 |
| question_other | cm_cka | 1.765 | 1.708 | 0.057 |
| question_other | cm_r2 | 1.671 | 2.065 | -0.394 |
| qa_concat | VSA | 1.615 | 1.622 | -0.007 |
| qa_concat | mutual_kNN_k5 | 1.710 | 1.745 | -0.035 |
| qa_concat | mutual_kNN_k10 | 1.772 | 1.897 | -0.125 |
| qa_concat | mutual_kNN_k20 | 1.755 | 1.817 | -0.062 |
| qa_concat | cm_cka | 1.635 | 1.622 | 0.013 |
| qa_concat | cm_r2 | 1.809 | 2.154 | -0.345 |
| answer_all_types | VSA | 1.704 | 1.681 | 0.023 |
| answer_all_types | mutual_kNN_k5 | 1.658 | 1.899 | -0.241 |
| answer_all_types | mutual_kNN_k10 | 1.728 | 1.910 | -0.181 |
| answer_all_types | mutual_kNN_k20 | 1.780 | 1.934 | -0.154 |
| answer_all_types | cm_cka | 1.720 | 1.767 | -0.047 |
| answer_all_types | cm_r2 | 1.668 | 2.024 | -0.356 |
| eval_answer | VSA | NA | NA | NA |
| eval_answer | mutual_kNN_k5 | NA | NA | NA |
| eval_answer | mutual_kNN_k10 | NA | NA | NA |
| eval_answer | mutual_kNN_k20 | NA | NA | NA |
| eval_answer | cm_cka | NA | NA | NA |
| eval_answer | cm_r2 | NA | NA | NA |

## Bug 修复审计

三个指标统一从 `configs/metrics.yaml` 读取方向；修前/修后三协议完整值见 `e4.json`。

| 域 | 指标 | 修前 regret | 修后 regret |
|---|---|---:|---:|
| caption | m50 | 12.313 | 12.332 |
| caption | m90 | 11.806 | 11.867 |
| caption | LAR_64 | 11.505 | 6.193 |
| answer | m50 | 12.794 | 12.887 |
| answer | m90 | 12.193 | 12.114 |
| answer | LAR_64 | 12.163 | 3.449 |

## 6. 附录：Lift/LAR

E3 的 Lift/LAR 失败结论保留，不再作为候选指标。完整三协议值已复制进 `e4.json`。
