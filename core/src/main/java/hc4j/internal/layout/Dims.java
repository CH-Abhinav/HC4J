/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package hc4j.internal.layout;

/**
 * Logical GEMM dimensions of op(a) · op(b).
 *
 * @param m rows of the result
 * @param n columns of the result
 * @param k reduction inner dimension (cols of op(a), rows of op(b))
 * @param resultShape final shape of the result tensor
 */
public record Dims(int m, int n, int k, int[] resultShape) {}
