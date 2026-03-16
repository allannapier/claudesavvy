class HistoryStack {
  constructor() {
    this._stack = [];
    this._pointer = -1;
  }

  push(htmlString) {
    // Truncate any redo states ahead of current pointer
    this._stack = this._stack.slice(0, this._pointer + 1);
    this._stack.push(htmlString);
    this._pointer = this._stack.length - 1;
  }

  undo() {
    if (!this.canUndo()) return null;
    this._pointer--;
    return this._stack[this._pointer];
  }

  redo() {
    if (!this.canRedo()) return null;
    this._pointer++;
    return this._stack[this._pointer];
  }

  current() {
    if (this._pointer < 0) return null;
    return this._stack[this._pointer];
  }

  canUndo() {
    return this._pointer > 0;
  }

  canRedo() {
    return this._pointer < this._stack.length - 1;
  }

  clear() {
    this._stack = [];
    this._pointer = -1;
  }
}
