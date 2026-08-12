/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/parsers.h — 参数解析器聚合头
 * ----------------------------------------------------------------------------
 * 仅用于把各类型 parser（default/enum/bitset/list）一次性包含进来，
 * 方便 param.h 统一引用，本身不含逻辑。
 */

/*
 * src/include/param/parsers.h — 参数解析器聚合头
 * ----------------------------------------------------------------------------
 * 仅用于把各类型 parser（default/enum/bitset/list）一次性包含进来，
 * 方便 param.h 统一引用，本身不含逻辑。
 */

#ifndef PARAM_PARSERS_H_INCLUDED
#define PARAM_PARSERS_H_INCLUDED

#include "param/parser_common.h"
#include "param/parser_default.h"
#include "param/parser_enum.h"
#include "param/parser_bitset.h"
#include "param/parser_list.h"

#endif /* PARAM_PARSERS_H_INCLUDED */
