/*
 * Create sidenotes
 */


$(function() {
    let _sidenoteCounter = 0
    document.querySelectorAll("span.sidenote").forEach(
        sidenote => {
            let snLabel = document.createElement("label");
            snLabel.setAttribute("for", `snid${_sidenoteCounter}`);
            snLabel.setAttribute("class", "margin-toggle sidenote-number");
            
            let snInput = document.createElement("input");
            snInput.setAttribute("type", "button");
            snInput.setAttribute("class", "margin-toggle");
            snInput.setAttribute("id", `snid${_sidenoteCounter}`);

            sidenote.before(snLabel, snInput)
            
            // sidenote.before(
            //     `<label for=\"snid${_sidenoteCounter}\" class=\"margin-toggle sidenote-number\"></label><input type=\"button\" class=\"margin-toggle\" id=\"snid${_sidenoteCounter}\"/>`
            // );
            _sidenoteCounter = _sidenoteCounter + 1;
	});
  });
  