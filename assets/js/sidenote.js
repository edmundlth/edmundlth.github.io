/*
 * Create sidenotes
 */


$(function() {
    /* create checked checkbox */
    var randomId = "snid" + Math.random().toString(16).slice(2);
    $("span[class=sidenote]").before(
        `<label for=\"${randomId}\" class=\"margin-toggle sidenote-number\"></label><input type=\"button\" class=\"margin-toggle\" id=\"${randomId}\"/>`
        );
  });
  