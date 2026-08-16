"""Учитель, чей forward считается маленькими кусками батча вместо одного
большого прогона — способ развязать batch_size, под который тюнились lr и
расписание оптимизатора (студент, criterion, шаг optimizer.step() не
меняются), от того, сколько элементов батча физически влезает в ОДИН
forward тяжёлого учителя за раз.

Зачем. У учителя нет градиентов (SegmentationTrainer вызывает его под
torch.no_grad(), teacher.requires_grad_(False)) — значит, чанкинг
батча математически ТОЖДЕСТВЕН одному большому forward'у: это не
приближение и не другой результат, а просто более экономный по памяти
способ посчитать тот же самый тензор (проверено также в
src/training/trainer.py::segmentation_evaluate, тем же приёмом с
micro_batch_size). Убрав памятное ограничение учителя из уравнения, можно
держать эффективный батч студента/оптимизатора таким же, каким его тюнили
(вместо того чтобы понижать trainer-batch_size целиком и вместе с ним —
негласно — lr, число эффективных шагов на эпоху и т.п.).

Особенно полезно вместе с NativeResolutionTeacher (см.
src/models/native_resolution_teacher.py): апсемпл крупного батча целиком до
родного разрешения учителя легко упирается либо в память, либо в лимит
CUDA-ядра на число элементов тензора (upsample_bilinear2d, проверено на
SegFormer-B5 при батче 24 на 1024x2048). Оборачивайте им ЛЮБОГО учителя
(обычного или уже обёрнутого NativeResolutionTeacher/MultiScaleInference) —
он ничего не знает о том, что внутри, только режет батч на входе и
склеивает выход.

НЕ регистрируйте этот класс в unwrap_model (src/models/feature_extractor.py),
даже для удобства с feature-based KD: FeatureExtractor кладёт в
self.features[layer] результат ПОСЛЕДНЕГО срабатывания хука, а при чанкинге
хук на тапе учителя сработает по разу на каждый кусок — и в фичах молча
останется только последний кусок батча, а не все карты. Без регистрации в
unwrap_model комбинация с feature-based KD падает явной ошибкой "слой не
найден" — и это правильное поведение, а не недоделка.
"""

import torch
from torch import nn


class ChunkedTeacher(nn.Module):
    """Прогоняет обёрнутую модель кусками по micro_batch_size, возвращает
    результат так, будто она видела батч целиком.

    Args:
        model: учитель (или уже обёрнутый NativeResolutionTeacher/
            MultiScaleInference) — что угодно, отдающее логиты
            [B, C, H, W] на вход [B, 3, H, W].
        micro_batch_size: размер одного куска. Последний кусок может быть
            меньше, если batch_size не делится нацело.
    """

    def __init__(self, model: nn.Module, micro_batch_size: int) -> None:
        super().__init__()
        if micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size должен быть > 0, получено {micro_batch_size}"
            )
        # module: то же имя атрибута, что у MultiScaleInference/
        # NativeResolutionTeacher — по нему unwrap_model снимает обёртку
        # (см. src/models/feature_extractor.py).
        self.module = model
        self.micro_batch_size = int(micro_batch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        if batch_size <= self.micro_batch_size:
            return self.module(images)

        chunks = [
            self.module(images[start : start + self.micro_batch_size])
            for start in range(0, batch_size, self.micro_batch_size)
        ]
        return torch.cat(chunks, dim=0)
