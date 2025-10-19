export default class Controls {
    constructor(ControlType) {
      this.ControlType = ControlType;
    }

    get controlTypes() {
      return {
        CONTROL_DIMS: this.ControlType.values.CONTROL_DIMS,
        CONTROL_TRAIN_LOSS_AND_PREDS: this.ControlType.values.CONTROL_TRAIN_LOSS_AND_PREDS,
        CONTROL_VAL_LOSS: this.ControlType.values.CONTROL_VAL_LOSS,
        CONTROL_BATCH_EXAMPLES: this.ControlType.values.CONTROL_BATCH_EXAMPLES,
        CONTROL_FULL_LOSS: this.ControlType.values.CONTROL_FULL_LOSS,
        CONTROL_NEXT_BATCH: this.ControlType.values.CONTROL_NEXT_BATCH,
        CONTROL_NEXT_DIMS: this.ControlType.values.CONTROL_NEXT_DIMS,
        CONTROL_CONFIG: this.ControlType.values.CONTROL_CONFIG,
        CONTROL_MESHGRID_RESULTS: this.ControlType.values.CONTROL_MESHGRID_RESULTS,
        CONTROL_SGD_STEP: this.ControlType.values.CONTROL_SGD_STEP,
        CONTROL_QUIT: this.ControlType.values.CONTROL_QUIT,
      };
    }

    getTypeName(typeValue) {
      const entry = Object.entries(this.ControlType.values).find(([_, value]) => value === typeValue);
      return entry ? entry[0] : "UNKNOWN_TYPE";
    }
  }
