/*
 * Create sidenotes
 */


$(function() {
    let _sidenoteCounter = 0
    document.querySelectorAll("span[sidenote]").forEach(
        sidenote => {
            let snLabel = document.createElement("label");
            snLabel.setAttribute("for", `snid${_sidenoteCounter}`);
            snLabel.setAttribute("class", "margin-toggle sidenote-number");
            
            let snInput = document.createElement("input");
            snInput.setAttribute("type", "button");
            snInput.setAttribute("class", "margin-toggle");
            snInput.setAttribute("id", `snid${_sidenoteCounter}`);

            sidenote.before(snLabel, snInput)

            _sidenoteCounter = _sidenoteCounter + 1;
	});
  });
  